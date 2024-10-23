"""Miscellaneous utilities for blimp"""
from typing import List, Union, Optional
import logging

from welford import Welford
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


def convert_array_dtype(arr, dtype, round_floats_if_necessary=False, copy=True):
    # check requested type is allowed
    allowed_types = np.sctypes["int"] + np.sctypes["uint"] + np.sctypes["float"]
    if not dtype in allowed_types:
        raise TypeError("``dtype`` = {dtype} is either not recognised or not allowed")

    # convert input to a numpy array
    if isinstance(arr, da.core.Array):
        arr = np.ndarray(arr)
    if not isinstance(arr, np.ndarray):
        raise TypeError("``arr`` is not a ``numpy.ndarray``")

    old_dtype = arr.dtype
    new_dtype = dtype
    if copy:
        new = np.copy(arr)
    else:
        new = arr

    if old_dtype == new_dtype:
        pass

    elif old_dtype in np.sctypes["uint"]:
        # convert from uint
        if np.can_cast(old_dtype, new_dtype):
            new = new.astype(new_dtype)
        else:
            raise TypeError(f"Cannot cast {old_dtype} to {new_dtype}")

    elif old_dtype in np.sctypes["int"]:
        # convert from int
        if np.can_cast(old_dtype, new_dtype):
            new = new.astype(new_dtype)
        else:
            raise TypeError(f"Cannot cast {old_dtype} to {new_dtype}")

    elif old_dtype in np.sctypes["float"]:
        # convert from float
        if not round_floats_if_necessary:
            if np.can_cast(old_dtype, new_dtype):
                new = new.astype(new_dtype)
            else:
                raise TypeError(f"Cannot cast {old_dtype} to {new_dtype}")
        else:
            # round to the nearest integer and represent using
            # the minimum scalar data type
            new = np.rint(new)
            min_dtype_based_on_rounded_values = np.promote_types(
                np.min_scalar_type(round(np.max(new))), np.min_scalar_type(round(np.min(new)))
            )
            new = new.astype(min_dtype_based_on_rounded_values)
            if np.can_cast(new.dtype, new_dtype):
                new = new.astype(new_dtype)
            else:
                raise ValueError(
                    f"Tried to convert array of ``float`` to an ``int`` or ``uint`` dtype"
                    + f" but cannot convert to requested dtype ({new_dtype}) because the"
                    + f" minimum data type necessary to hold the values is ({new.dtype}) "
                )
    return new


def equal_dims(a, b, dimensions="TCZYX"):
    if not hasattr(a, "dims"):
        raise AttributeError(f"``a`` has no ``dims`` attribute")
    if not hasattr(b, "dims"):
        raise AttributeError(f"``b`` has no ``dims`` attribute")
    equal_T = a.dims.T == b.dims.T
    equal_C = a.dims.C == b.dims.C
    equal_Z = a.dims.Z == b.dims.Z
    equal_Y = a.dims.Y == b.dims.Y
    equal_X = a.dims.X == b.dims.X
    all_equal = True
    if "T" in dimensions:
        all_equal = all([all_equal, equal_T])
    if "C" in dimensions:
        all_equal = all([all_equal, equal_C])
    if "Z" in dimensions:
        all_equal = all([all_equal, equal_Z])
    if "Y" in dimensions:
        all_equal = all([all_equal, equal_Y])
    if "X" in dimensions:
        all_equal = all([all_equal, equal_X])
    return all_equal


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
    images: Union[AICSImage, List[AICSImage]], omit: Union[int, str, None] = None, check_dtype: bool = True,
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
    check_dtype
        True if images in list must have the same dtype

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
                if check_dtype:
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


def change_image_dtype(
    image: AICSImage,
    dtype: np.dtype
) -> AICSImage:
    """
    Change the dtype of an AICSImage object.

    This function converts the data type of the image's data using the 
    `convert_array_dtype` function. It preserves the channel 
    names and physical pixel sizes of the original image.

    Parameters
    ----------
    image : AICSImage
        The input AICSImage object containing the image data.
    dtype : np.dtype
        The target dtype to convert the image data to.

    Returns
    -------
    AICSImage
        A new AICSImage object with the updated dtype, 
        retaining the original channel names and physical pixel sizes.
    """
    return AICSImage(convert_array_dtype(image.data, dtype),
                     channel_names=image.channel_names,
                     physical_pixel_sizes=image.physical_pixel_sizes)


def mean_std_welford(images: List[AICSImage], log_transform: bool = False) -> tuple:
    n_c = images[0].dims.C
    w = [Welford() for i in range(n_c)]
    for image in images:
        for c in range(n_c):
            # use lazy loading to see if this fixes the memory leak
            array_lazy = image.get_image_dask_data("YX", C=c, T=0, Z=0)
            array = array_lazy.compute()
            if log_transform:
                is_zero = array == 0
                if np.any(is_zero):
                    logger.warn("image contains zero values")
                array = np.log10(array)
                # The log10 transform sets zero pixel values to -inf
                array[is_zero] = 0
            if np.any(np.isinf(array)):
                logger.warn("skip image because it contains infinite values")
            else:
                w[c].add(array)

    mean_arrays = [w[c].mean for c in range(n_c)]
    std_arrays = [np.sqrt(w[c].var_s) for c in range(n_c)]

    m = AICSImage(
        np.stack(mean_arrays, axis=0)[np.newaxis, :, np.newaxis, :, :],
        channel_names=images[0].channel_names,
        physical_pixel_sizes=images[0].physical_pixel_sizes,
    )
    s = AICSImage(
        np.stack(std_arrays, axis=0)[np.newaxis, :, np.newaxis, :, :],
        channel_names=images[0].channel_names,
        physical_pixel_sizes=images[0].physical_pixel_sizes,
    )

    return m, s


def average_images(images: List[AICSImage], keep_same_type: bool = True) -> AICSImage:
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
            np.mean([image.get_image_data("TZYX", C=channel) for image in images], axis=0)
            for channel in range(0, images[0].dims.C)
        ]
        arr = np.stack(channel_averages, axis=1)

    if keep_same_type:
        if images[0].dtype.kind in ["i", "u"]:
            # round to nearest integer
            arr = np.rint(arr).astype(images[0].dtype)
        else:
            arr = arr.astype(images[0].dtype)

    return AICSImage(arr, channel_names=images[0].channel_names, physical_pixel_sizes=images[0].physical_pixel_sizes)


def std_images(images: List[AICSImage], keep_same_type: bool = True) -> AICSImage:
    """
    Calculate the standard deviation (per pixel) of multiple images.

    Parameters
    ----------
    image_list
        A list of `AICSImage` objects.

    Returns
    -------
    AICSImage
        The standard deviation image.

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
        channel_stds = [
            np.std([image.get_image_data("TZYX", C=channel) for image in images], axis=0)
            for channel in range(0, images[0].dims.C)
        ]
        arr = np.stack(channel_stds, axis=1)

    if keep_same_type:
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

    channel_names = images[0].channel_names
    physical_pixel_sizes = images[0].physical_pixel_sizes
    dtype = images[0].dtype

    if order == "append":
        logger.debug(f"Concatenating by appending, on axis {axis}")
        arr = np.concatenate([image.get_image_dask_data("TCZYX") for image in images], axis=axis_int)
        if axis_int == 1:
            channel_names = [channel for image in images for channel in image.channel_names]

    elif order == "interleave":
        logger.debug(f"Concatenating by interleaving, on axis {axis}")

        n_times = images[0].dims.T
        n_planes = images[0].dims.Z

        # AICS order is 'TCZYX'
        if axis_int == 0:
            # interleave time
            arr = np.concatenate(
                [
                    np.concatenate([image.get_image_dask_data("TCZYX", T=time) for image in images], axis=0)
                    for time in range(n_times)
                ],
                axis=0,
            )

        if axis_int == 1:
            raise NotImplementedError("Interleaving of channels is not implemented. Try with ``order='append'``")

        if axis_int == 2:
            # interleave z_planes
            arr = np.concatenate(
                [
                    np.concatenate([image.get_image_dask_data("TCZYX", Z=z_plane) for image in images], axis=2)
                    for z_plane in range(n_planes)
                ],
                axis=2,
            )

    return AICSImage(arr, channel_names=channel_names, physical_pixel_sizes=physical_pixel_sizes)


def translate_array(img: np.ndarray, y: int, x: int):
    """Shifts an image by y and x pixels (pads with zeros)

    Parameters
    ----------
    img
        image that should be shifted
    y
        shift in y direction (positive value -> down, negative value -> up)
    x
        shift in x direction (position value -> right, negative value -> left)

    Returns
    -------
    numpy.array
        potentially shifted and cropped image

    """
    if y < 0:
        shifted_img = np.pad(img, ((0, abs(y)), (0, 0)), mode="constant")[abs(y) :, :]
    elif y > 0:
        shifted_img = np.pad(img, ((y, 0), (0, 0)), mode="constant")[:-y, :]
    elif y == 0:
        shifted_img = np.ndarray.copy(img)

    if x < 0:
        shifted_img = np.pad(shifted_img, ((0, 0), (0, abs(x))), mode="constant")[:, abs(x) :]
    elif x > 0:
        shifted_img = np.pad(shifted_img, ((0, 0), (x, 0)), mode="constant")[:, :-x]
    elif x == 0:
        pass

    return shifted_img


def _vollath_f4(arr: np.ndarray):
    """Vollath's F4 function.

    Described in https://doi.org/10.1002/jemt.24035

    Code adapted from
    https://github.com/aquimb/Autofocus-URJC/blob/d3b83382a82315e27e248d09519e3da4dbac40f6/Code/algorithms.py#L200
    """
    sum1 = 0
    sum2 = 0
    rows = arr.shape[0]
    cols = arr.shape[1]
    for i in range(rows - 1):
        sum1 = sum1 + (arr[i] * arr[i + 1])

    for i in range(rows - 2):
        sum2 = sum2 + (arr[i] * arr[i + 2])

    sum3 = sum1 - sum2
    res = 0
    for i in range(cols):
        res = res + sum3[i]  # type: ignore

    return res


def estimate_focus_plane(
    image: AICSImage, C: Optional[int], sliding_window: Optional[int] = None, crop: float = 1.0
) -> int:
    """
    Estimate the focus plane of a given image using Vollath's F4 method.

    (Vollath's F4 is highest at the focus plane)

    Parameters
    ----------
    image
        The input image.
    C
        The channel to use for the calculation, by default 0.
    sliding_window
        The size of the sliding window to average the result, by default None.
    crop
        The amount of cropping to apply to the image, by default 1.0.

    Returns
    -------
    int
        The estimated focus plane.

    Raises
    ------
    TypeError
        If `crop` is not of type float.
    ValueError
        If `crop` is not less than 1.
    """

    if not isinstance(crop, float):
        raise TypeError("``crop`` must be of type float")
    if not isinstance(crop, AICSImage):
        raise TypeError("``image`` must be an ``aicsimageio.AICSImage``")

    if C is None:
        logger.warning("Using default channel (C = 0) to calculate focus_plane")
        C = 0

    arrays = [image.get_image_data("YX", T=0, C=C, Z=i) for i in range(image.dims.Z)]
    if crop < 1.0:
        logger.debug(f"cropping image with factor {crop}")
        ymin = np.floor((1.0 - crop) * arrays[0].shape[0]).astype(int)
        ymax = np.ceil(crop * arrays[0].shape[0]).astype(int)
        xmin = np.floor((1.0 - crop) * arrays[0].shape[1]).astype(int)
        xmax = np.ceil(crop * arrays[0].shape[1]).astype(int)
        arrays = [a[ymin:ymax, xmin:xmax] for a in arrays]
    elif crop != 1.0:
        raise ValueError("``crop`` must be <= 1")

    f4 = [_vollath_f4(arr) for arr in arrays]
    # If necessary, could also use the intensity to remove
    # out-of-focus images from stack
    # e.g. intensity = [np.mean(arr) for arr in arrays]

    if sliding_window is not None:
        if sliding_window > 0:
            logger.debug(f"cropping image with window size {crop}")
            # average using sliding window
            f4_avg = list(np.convolve(f4, np.ones(sliding_window) / sliding_window, mode="valid"))
            # pad with zeros
            if sliding_window % 2 == 0:
                pad_left = list(np.zeros([np.floor(float(sliding_window - 1) / 2.0).astype(int)], dtype=type(f4[0])))
            else:
                pad_left = list(np.zeros([np.floor(float(sliding_window) / 2.0).astype(int)], dtype=type(f4[0])))
            pad_right = list(np.zeros([np.floor(float(sliding_window) / 2.0).astype(int)], dtype=type(f4[0])))
            f4 = pad_left + f4_avg + pad_right
            # find position of max vollath_f4 (this is the focus plane)
            max_pos = np.argmax(f4_avg) + len(pad_left)
        else:
            max_pos = np.argmax(f4)

    return int(max_pos)
