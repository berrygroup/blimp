"""Miscellaneous tools for blimp"""
from typing import List, Union
import logging

from aicsimageio import AICSImage
import numpy as np
import dask.array as da

logger = logging.getLogger(__name__)

AXIS_STR_TO_INT = {
    "T": 0,
    "C": 1,
    "Z": 2,
    "Y": 3,
    "X": 4,
    "t": 0,
    "c": 1,
    "z": 2,
    "y": 3,
    "x": 4,
}

AXIS_INT_TO_STR = {0: "T", 1: "C", 2: "Z", 3: "Y", 4: "X"}


def _axis_str_to_int(axis: Union[str, int]) -> int:
    """Convert AICSImage dimension character to corresponding int."""
    if isinstance(axis, str):
        if axis not in AXIS_STR_TO_INT.keys():
            raise ValueError(f"Unknown axis : {axis}")
        else:
            axis = AXIS_STR_TO_INT[axis]
    elif isinstance(axis, int):
        if axis not in AXIS_INT_TO_STR.keys():
            raise ValueError(f"Unknown axis : {axis}")
    else:
        raise TypeError("axis must be int or str")

    return axis


def _axis_int_to_str(axis: Union[str, int]) -> str:
    """Convert int to corresponding AICSImage dimension character."""
    if isinstance(axis, int):
        if axis not in AXIS_INT_TO_STR.keys():
            raise ValueError(f"Unknown axis : {axis}")
        else:
            axis = AXIS_INT_TO_STR[axis]
    elif isinstance(axis, str):
        if axis not in AXIS_STR_TO_INT.keys():
            raise ValueError(f"Unknown axis : {axis}")
    else:
        raise TypeError("axis must be int or str")

    return axis


def confirm_array_rank(
    images: Union[np.ndarray, List[np.ndarray], da.core.Array, List[da.core.Array]], rank: int = 3
) -> None:
    """
    Check the rank of array(s) and confirm that it matches the required rank.

    Parameters:
    images (Union[np.ndarray, List[np.ndarray], da.core.Array, List[da.core.Array]]): Image(s) to be checked.
    rank (int, optional): Required rank of the image(s). Default is 3.

    Returns:
    None

    Raises:
    TypeError
        If input is not of type numpy.ndarray, dask.array.core.Array or list of such objects.
    ValueError
        If input does not have the correct rank.
    """
    if isinstance(images, np.ndarray) or isinstance(images, da.core.Array):
        if len(images.shape) != rank:
            raise ValueError(f"Input does not have correct rank (shape = {images.shape})")

    elif isinstance(images, list):
        for image in images:
            if not (isinstance(image, np.ndarray) or isinstance(image, da.core.Array)):
                raise TypeError("Input should be of type numpy.ndarray, dask.array.core.Array or list of such objects")
            elif len(image.shape) != rank:
                raise ValueError(f"Input does not have correct rank (shape = {image.shape})")
    else:
        raise TypeError("Input should be of type numpy.ndarray, dask.array.core.Array or list of such objects")
    return None


def read_template(template_name: str) -> str:
    import importlib.resources as pkg_resources

    from . import templates

    return pkg_resources.read_text(templates, template_name)


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


def check_uniform_dimension_sizes(
    images: Union[AICSImage, List[AICSImage]], omit: Union[int, str, None] = None
) -> bool:
    """
    Check that the dimensions of a list of AICSImages are uniform.

    Parameters
    ----------
    images
        A single AICSImage or a list of AICSImages to check for matching
        dimension sizes.
    omit
        Integer or string designating a single axis to be omitted
        from checking (e.g. 'T' or 0 for time axis, or 'C' or 1 for
        channel axis. Order is 'TCZYX'). Default = None

    Returns
    -------
    bool
        True if all images in the list have matching dimension sizes, False
        otherwise.

    Raises
    ------
    TypeError
        If input is not an AICSImage or list of AICSImages.
    TypeError
        If dtypes are not the same for all elements of the list.
    """

    # first check that the dimension orders match
    try:
        correct_dimension_order = check_correct_dimension_order(images)
    except TypeError:
        raise TypeError("Incorrect input type")

    if isinstance(omit, (str, int)):
        omit = _axis_str_to_int(omit)

    # then check that the dimension sizes match
    if correct_dimension_order:
        if isinstance(images, list):
            if all([isinstance(el, AICSImage) for el in images]):
                # check that dtypes are the same
                first_dtype = images[0].dtype
                if not all([image.dtype == first_dtype for image in images]):
                    raise TypeError("List of AICSImages have non-matching data types")
                # compare sizes
                dim_sizes = [image.dims.shape for image in images]
                first = dim_sizes[0]
                if omit is None:
                    # check all dimensions
                    result = all([ds == first for ds in dim_sizes])
                else:
                    # check all dimensions, except 'omit'
                    # Note: list[:x] + list[x+1:] = 'list without element x'
                    result = all(
                        [(ds[:omit] + ds[omit + 1 :]) == (first[:omit] + first[omit + 1 :]) for ds in dim_sizes]
                    )
        elif isinstance(images, AICSImage):
            # only one image passed
            result = True
        else:
            raise TypeError("Input is not an AICSImage or list of AICSImages")
    else:
        logger.error("Not all AICSImages in list have the same dimension order")
        result = False

    return result


def average_images(images: List[AICSImage]) -> AICSImage:
    """
    Calculate the average of multiple images.

    Parameters
    ----------
    image_list
        A list of `AICSImage` objects to be averaged.

    Returns
    -------
    AICSImage
        The averaged image.

    Raise
    ------
    TypeError
        if elements of image_list have incorrect types
    ValueError
        if images in image_list have different dimensions

    Notes
    -----
    This function first checks that all images in the list have matching dimension
    sizes using the `check_uniform_dimension_sizes()` function. If the dimensions
    do not match, an error message is logged and the program exits. Otherwise, the
    function calculates the mean of all images for each channel and stacks the
    resulting arrays together along the `C` axis, returning the final average
    image as a numpy array.
    """
    try:
        dimension_sizes_match = check_uniform_dimension_sizes(images)
    except TypeError:
        raise TypeError("Cannot average over list elements of different types")

    if not dimension_sizes_match:
        raise ValueError("Cannot average over images of different sizes")
    else:
        channel_averages = [
            np.mean([image.get_image_dask_data("TZYX", C=channel) for image in images], axis=0)
            for channel in range(0, images[0].dims.C)
        ]
        arr = np.stack(channel_averages, axis=1)

    if images[0].dtype.kind in ["i", "u"]:
        # round to nearest integer
        arr = np.rint(arr).astype(images[0].dtype)
    else:
        arr = arr.astype(images[0].dtype)

    return AICSImage(arr, channel_names=images[0].channel_names, physical_pixel_sizes=images[0].physical_pixel_sizes)


def concatenate_images(
    images: Union[AICSImage, List[AICSImage]], axis: Union[int, str] = 0, order: str = "interleave"
) -> AICSImage:
    """
    Concatenates multiple `AICSImage` instances along specified dimensions.

    Parameters
    ----------
    images
        A list of `AICSImage` instances to be concatenated.
    axis
        An integer or character representing the axis to concatenate
        (e.g. 'T' or 0 for time axis, or 'C' or 1 for channel axis. Order is
        'TCZYX'). Only 'T','C','Z' are implemented. Default = T.
    order
        Either 'interleave' or 'append'. Determines whether the images in the
        list are appended to along the current axis or interleaved.

    Returns
    -------
    AICSImage
        The concatenated images

    Raises
    ------
    TypeError
        If the sizes of the dimensions of the images passed in the `images`
        argument do not match.

    Notes
    -----
    If a single `AICSImage` instance is passed as the `images` argument,
    the function will return that single image.
    """
    axis_int = _axis_str_to_int(axis)
    try:
        dimension_sizes_match = check_uniform_dimension_sizes(images, omit=axis_int)
    except TypeError:
        raise TypeError("Cannot concatenate list elements of different types")

    if isinstance(images, AICSImage):
        logger.warn("Tried to concatenate a single image, result is unchanged")
        return images

    elif not dimension_sizes_match:
        raise TypeError("Cannot concatenate images of different sizes")

    if order == "append":
        logger.debug(f"Concatenating by appending, on axis {axis}")
        arr = np.concatenate([image.get_image_dask_data("TCZYX") for image in images], axis=axis_int)
    elif order == "interleave":
        logger.debug(f"Concatenating by interleaving, on axis {axis}")

        n_times = images[0].dims.T
        n_channels = images[0].dims.C
        n_planes = images[0].dims.Z

        # AICS order is 'TCZYX'
        if axis == 0:
            # interleave time
            arr = np.concatenate(
                [
                    np.concatenate([image.get_image_dask_data("TCZYX", T=time) for image in images], axis=0)
                    for time in range(n_times)
                ],
                axis=0,
            )

        if axis == 1:
            # interleave channels
            arr = np.concatenate(
                [
                    np.concatenate([image.get_image_dask_data("TCZYX", C=channel) for image in images], axis=1)
                    for channel in range(n_channels)
                ],
                axis=1,
            )

        if axis == 2:
            # interleave z_planes
            arr = np.concatenate(
                [
                    np.concatenate([image.get_image_dask_data("TCZYX", Z=z_plane) for image in images], axis=2)
                    for z_plane in range(n_planes)
                ],
                axis=2,
            )

    if images[0].dtype.kind in ["i", "u"]:
        # round to nearest integer
        arr = np.rint(arr).astype(images[0].dtype)
    else:
        arr = arr.astype(images[0].dtype)

    return AICSImage(arr, channel_names=images[0].channel_names, physical_pixel_sizes=images[0].physical_pixel_sizes)
