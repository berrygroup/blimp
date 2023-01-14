"""Miscellaneous tools for blimp."""
from typing import List, Union
import os
import logging

from aicsimageio import AICSImage
import numpy as np

logger = logging.getLogger(__name__)


def check_correct_dimension_order(images: Union[AICSImage, List[AICSImage]]) -> bool:
    """
    Check that the order of dimensions is 'TCZYX'.

    Parameters
    ----------
    images
        A single `AICSImage` or a list of `AICSImage`s to check the dimension order of.

    Returns
    -------
    bool
        True if the dimension order is TCZYX for all images, False otherwise.

    Raises
    ------
    TypeError
        If input is not an AICSImage or a list of AICSImages.
        If input is a list and not all elements are of type AICSImage.
    """
    if isinstance(images, list):
        if all([isinstance(el, AICSImage) for el in images]):
            dim_orders = [image.dims.order for image in images]
            result = all([element == "TCZYX" for element in dim_orders])
        else:
            raise TypeError("Not all list elements are of type AICSImage")
            result = False
    elif isinstance(images, AICSImage):
        result = images.dims.order == "TCZYX"
    else:
        raise TypeError("Input is not an AICSImage or list of AICSImages")
        result = False

    return result


def check_correct_dimension_sizes(images: Union[AICSImage, List[AICSImage]]) -> bool:
    """
    Check that the dimensions of a list of AICSImages are correct and uniform.

    Parameters
    ----------
    images
        A single AICSImage or a list of AICSImages to check for correct and
        matching dimension sizes.

    Returns
    -------
    bool
        True if all images in the list have matching dimension sizes, False
        otherwise.

    Raises
    ------
    TypeError: If input is not an AICSImage or list of AICSImages.
    """

    # first check that the dimension orders match
    try:
        correct_dimension_order = check_correct_dimension_order(images)
    except TypeError:
        logger.error("Incorrect input type")

    # then check that the dimension sizes match
    if correct_dimension_order:
        if isinstance(images, list):
            if all([isinstance(el, AICSImage) for el in images]):
                dim_sizes = [image.dims.shape for image in images]
                result = all([element == dim_sizes[0] for element in dim_sizes])
        elif isinstance(images, AICSImage):
            # only one image passed
            result = True
        else:
            raise TypeError("Input is not an AICSImage or list of AICSImages")
            result = False
    else:
        logger.error("Not all AICSImages in list have the same dimension order")
        result = False

    return result


def average_images(image_list: List[AICSImage]) -> np.ndarray:
    """
    Calculate the average of multiple images.

    Parameters
    ----------
    image_list : List[AICSImage]
        A list of `AICSImage` objects to be averaged.

    Returns
    -------
    np.ndarray
        The average image as a numpy array.

    Raise
    ------
    error : if the image_list have different image dimension

    Notes
    -----
    This function first checks that all images in the list have matching dimension
    sizes using the `check_correct_dimension_sizes()` function. If the dimensions
    do not match, an error message is logged and the program exits. Otherwise, the
    function calculates the mean of all images for each channel and stacks the
    resulting arrays together along the `C` axis, returning the final average
    image as a numpy array.
    """
    try:
        dimension_sizes_match = check_correct_dimension_sizes(image_list)
    except TypeError:
        logger.error("fatal: cannot average over list elements of different types")
        os._exit(1)

    if not dimension_sizes_match:
        logger.error("fatal: cannot average over images of different sizes")
        os._exit(1)
    else:
        channel_averages = [
            np.mean([image.get_image_dask_data("TZYX", C=channel) for image in image_list], axis=0)
            for channel in range(0, image_list[0].dims.C)
        ]
        result = np.stack(channel_averages, axis=1)

    return result


def concatenate_images(
    images: Union[AICSImage, List[AICSImage]], output_order: str = "CZYX"
) -> Union[np.ndarray, None]:
    """
    Concatenates multiple `AICSImage` instances along specified dimensions.

    Parameters
    ----------
    images
        A list of `AICSImage` instances to be concatenated.
    output_order
        The order in which the images should be concatenated.
        Only 'CZYX' is currently supported.

    Returns
    -------
    Union[np.ndarray,None]
        A numpy array containing the concatenated images. If the `output_order`
        parameter is not 'CZYX', the function returns None.

    Raises
    ------
    TypeError
        If the sizes of the dimensions of the images passed in the `images`
        argument do not match.
    NotImplementedError
        If the `output_order` parameter is not 'CZYX'.

    Notes
    -----
    If a single `AICSImage` instance is passed as the `images` argument, the function will return that single image.
    """
    try:
        dimension_sizes_match = check_correct_dimension_sizes(images)
    except TypeError:
        logger.error("fatal: cannot average over list elements of different types")
        os._exit(1)

    if isinstance(images, AICSImage):
        logger.warn("Single image concatenated")
        result = images
    elif not dimension_sizes_match:
        logger.error("fatal: do not concatenate images of different sizes")
        os._exit(1)
    elif output_order == "CZYX":
        logger.debug("Concatenating images with order 'CZYX'")
        # concatenate all sites at each timepoint
        # then iterate over timepoints
        channel_images = [
            np.concatenate(
                [
                    np.stack([image.get_image_dask_data("ZYX", C=channel, T=time) for image in images], axis=0)
                    for time in range(images[0].dims.T)
                ]
            )
            for channel in range(0, images[0].dims.C)
        ]

        result = np.stack(channel_images, axis=1)
    else:
        logger.debug(f"output order {output_order}")
        raise NotImplementedError(f"output_order = {output_order} not implemented")
        result = None

    return result
