# pip install itk-elastix
from typing import List, Tuple, Union, Literal, overload
from numbers import Integral
from pathlib import Path
import logging

from itk import ParameterObject, transformix_filter, elastix_registration_method
from aicsimageio import AICSImage
import numpy as np
import dask.array as da
import image_registration as reg

from blimp.utils import (
    translate_array,
    confirm_array_rank,
    check_uniform_dimension_sizes,
)

logger = logging.getLogger(__name__)


class TransformationParameters:
    def __init__(
        self,
        transformation_mode: Union[str, None] = None,
        resolutions: int = 3,
        from_object: Union[ParameterObject, None] = None,
        from_file: Union[Path, str, None] = None,
        from_resources: Union[str, None] = None,
    ):
        """
        Class for use with the image registration functions using ITK elastix.

        The class has the following attributes:
        transformation_mode:
            To initialise default transformation parameters, pass one of the
            following strings (representing the mode of transformation):
            'translation', 'rigid' or 'affine'. Default is None.
        resolutions:
            When specifying the ``transformation mode``, resolutions can
            be used to tweak the the number of resolutions. Default is 3.
        from_object:
            To initialise from an existing itk.ParameterObject, pass the object
            as ``from_object``. Default is None.
        from_file:
            To initialise from a file, pass the filename as ``from_file``.
            Default is None.
        from_resources:
            To read default parameters from the BLIMP package. pass one of
            the following strings as ``from resources``. Default is None.

        The init method initializes the class attributes and initialises a
        itk.ParameterObject in the ``parameter_object`` attribute, based on the
        initialisation parameters.

        The save method writes the current parameter_object to a file
        specified in the file_name parameter. This file can be loaded using
        ``TranformationParameters(from_file = {file_name})``
        """
        self.from_file = str(from_file)
        self.from_resources = str(from_file)
        self.transformation_mode = transformation_mode
        self.from_object = True if from_object is not None else False

        if all([transformation_mode, from_resources, from_file, from_object]) is None:
            raise ValueError(
                "TransformationParameters requires one of [transformation_mode, from_object, from_file, from_resources]"
            )

        if from_object is not None:
            logger.debug("Initialising ``TransformationParameters`` from object")
            self.parameter_object = from_object

        elif from_file is not None:
            logger.debug(f"Initialising ``TransformationParameters`` from file {from_file}")
            self.parameter_object = ParameterObject.New()
            self.parameter_object.AddParameterFile(str(from_file))

        elif from_resources is not None:
            logger.debug(f"Initialising ``TransformationParameters`` from resources {from_resources}")
            # TODO: implement default transformation parameters in resources
            # load_transformation_parameters_from_resources(from_resources)
            # self.parameter_object = ParameterObject.New()
            raise NotImplementedError("Default transformation settings in package resources are not implemented yet")

        else:
            logger.debug(f"Initialising ``TransformationParameters`` using defaults: {transformation_mode}")
            self.parameter_object = ParameterObject.New()
            if transformation_mode in ["translation", "rigid", "affine"]:
                parameter_map = self.parameter_object.GetDefaultParameterMap(transformation_mode, resolutions)
            else:
                logger.error("Unknown transformation mode {transformation_mode}")
                raise ValueError(f"ITK transformation_mode {transformation_mode} unknown")

            self.parameter_object.AddParameterMap(parameter_map)

    def save(self, file_name: Union[str, Path]):
        logger.debug("Saving transformation parameters as {file_name}")
        ParameterObject.WriteParameterFile(self.parameter_object, str(file_name))

    def __str__(self):
        return self.parameter_object.__str__()


@overload
def register_2D(
    fixed: Union[np.ndarray, da.core.Array],
    moving: Union[np.ndarray, da.core.Array],
    settings: TransformationParameters,
    parameters_only: Literal[False] = ...,
) -> Tuple[np.ndarray, TransformationParameters]:
    ...


@overload
def register_2D(
    fixed: Union[np.ndarray, da.core.Array],
    moving: Union[np.ndarray, da.core.Array],
    settings: TransformationParameters,
    parameters_only: Literal[True],
) -> TransformationParameters:
    ...


def register_2D(
    fixed: Union[np.ndarray, da.core.Array],
    moving: Union[np.ndarray, da.core.Array],
    settings: TransformationParameters,
    parameters_only: bool = False,
) -> Union[Tuple[np.ndarray, TransformationParameters], TransformationParameters]:
    """Align an image to a reference image, keeping transformation parameters.

    Aligns a ``moving`` image to a ``fixed`` image using ITK Elastix.

    Parameters
    ----------
    fixed
        a reference image
    moving
        an image to be aligned to the reference image
    settings
        object of class TransformationParameters to provide
        the initial setting for performing the registration
        See ``blimp.utils.TransformationParameters`` for
        more details.

    Returns
    -------
    numpy.ndarray
        registered image derived by transforming ``moving``
        to the reference frame of ``fixed``.
    TransformationParameters
        object containing the full transformation parameters
        necessary to perform the transformation of ``moving``
        to the reference frame of ``fixed``.

    Raises
    ------
    TypeError
        If any of the positional inputs are not of the correct type
        If the numpy.dtypes of the two input arrays do not match
    ValueError
        If the input arrays are not rank 2
        If the shape of input arrays does not match
    """

    try:
        confirm_array_rank([fixed, moving], 2)
    except TypeError:
        raise TypeError(
            "Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray or dask.array.core.Array)"
        )
    except ValueError:
        raise ValueError("Either ``fixed`` or ``moving`` are not of correct rank (2)")

    if fixed.shape != moving.shape:
        raise ValueError(f"Shape of ``moving`` ({fixed.shape}) must match shape of ``fixed`` ({moving.shape})")
    elif fixed.dtype != moving.dtype:
        raise TypeError(
            f"numpy.dtype of ``moving`` ({fixed.dtype}) must match numpy.dtype of ``fixed`` ({moving.dtype})"
        )
    elif not isinstance(settings, TransformationParameters):
        raise TypeError("``settings`` is not a ``TransformationParameters`` object")

    registered, parameters = elastix_registration_method(
        np.asarray(fixed, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        parameter_object=settings.parameter_object,
    )
    if parameters_only:
        return TransformationParameters(from_object=parameters)
    else:
        return (np.asarray(registered, dtype=moving.dtype), TransformationParameters(from_object=parameters))


def transform_2D(moving: Union[np.ndarray, da.core.Array], parameters: TransformationParameters) -> np.ndarray:
    """Transform an image using a defined parameters.

    Apply set of transformation parameters to an input
    image using ITK Elastix.

    Parameters
    ----------
    moving
        an image to be transformed
    parameters
        object of class TransformationParameters to provide
        the full parameterisation of the transformation.
        See ``blimp.utils.TransformationParameters`` for
        more details.

    Returns
    -------
    numpy.ndarray
        transformed image derived by transforming ``moving``

    Raises
    ------
    TypeError
        If any of the positional inputs are not of the correct type
    ValueError
        If the input array is not rank 2
    RuntimeError
        If the shape specified in ``parameters`` does not match
        the shape of ``moving``
    """

    try:
        confirm_array_rank(moving, 2)
    except TypeError:
        raise TypeError("``moving`` is not of correct type (numpy.ndarray)")
    except ValueError:
        raise ValueError("``moving`` is not of correct rank (2)")

    if not isinstance(parameters, TransformationParameters):
        raise TypeError("``settings`` is not a ``TransformationParameters`` object")
    else:
        x, y = (int(float(i)) for i in parameters.parameter_object.GetParameterMap(0)["Size"])
        if not (y == moving.shape[0] and x == moving.shape[1]):
            raise RuntimeError(
                f"`TransformationParameters` expected size ({y},{x}), "
                + f"while ``moving`` has size ({moving.shape[0]},{moving.shape[1]})"
            )

    transformed = transformix_filter(np.asarray(moving, dtype=np.float32), parameters.parameter_object)

    return np.asarray(transformed, dtype=moving.dtype)


def _recast_array(
    arr: np.ndarray,
    dtype: np.dtype,
    remove_negative: bool = False,
) -> np.ndarray:

    # if desired dtype is integer, round the values
    if issubclass(dtype.type, Integral):
        arr = np.round(arr)

    # remove negative values
    if remove_negative:
        arr[arr < 0] = 0

    return arr.astype(dtype.type)


@overload
def register_2D_fast(
    fixed: Union[np.ndarray, da.core.Array],
    moving: Union[np.ndarray, da.core.Array],
    parameters_only: Literal[False] = ...,
) -> Tuple[np.ndarray, Tuple]:
    ...


@overload
def register_2D_fast(
    fixed: Union[np.ndarray, da.core.Array],
    moving: Union[np.ndarray, da.core.Array],
    parameters_only: Literal[True],
) -> Tuple:
    ...


def register_2D_fast(
    fixed: Union[np.ndarray, da.core.Array], moving: Union[np.ndarray, da.core.Array], parameters_only: bool = False
) -> Union[Tuple[np.ndarray, Tuple], Tuple]:
    """Align an image to a reference image, keeping transformation parameters.

    This approach uses the ``image_registration`` package,
    which is faster than ITK elastix. It uses Fourier
    transformation and chi2 to calculate x-y shifts. It is
    very simple and may be sufficient for some multi-cycle
    experiments.

    Parameters
    ----------
    fixed
        a reference image
    moving
        an image to be aligned to the reference image

    Returns
    -------
    numpy.ndarray
        registered image derived by transforming ``moving``
        to the reference frame of ``fixed``. Omitted with
        ``parameters_only`` is True.
    tuple
        (xoff,yoff), the x and y offsets required for
        registration of ``moving`` to ``fixed``.

    Raises
    ------
    TypeError
        If any of the positional inputs are not of the correct type
        If the numpy.dtypes of the two input arrays do not match
    ValueError
        If the input arrays are not rank 2
        If the shape of input arrays does not match
    """

    try:
        confirm_array_rank([fixed, moving], 2)
    except TypeError:
        raise TypeError(
            "Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray or dask.array.core.Array)"
        )
    except ValueError:
        raise ValueError("Either ``fixed`` or ``moving`` are not of correct rank (2)")

    if fixed.shape != moving.shape:
        raise ValueError(f"Shape of ``moving`` ({fixed.shape}) must match shape of ``fixed`` ({moving.shape})")
    elif fixed.dtype != moving.dtype:
        raise TypeError(
            f"numpy.dtype of ``moving`` ({fixed.dtype}) must match numpy.dtype of ``fixed`` ({moving.dtype})"
        )

    # image_registration does not play nice with dask arrays,
    # compute these here to convert to numpy.ndarray
    if isinstance(fixed, da.core.Array):
        fixed = fixed.compute()
    if isinstance(moving, da.core.Array):
        moving = moving.compute()

    xoff, yoff, exoff, eyoff = reg.chi2_shift(fixed, moving)
    # subpixel fitting using the following function
    # results in negative values and FFT artefacts
    # registered = reg.fft_tools.shift.shiftnd(moving, (-yoff, -xoff))
    xoff = int(np.round(xoff))
    yoff = int(np.round(yoff))
    registered = translate_array(moving, -yoff, -xoff)

    if parameters_only:
        return (xoff, yoff)
    else:
        return (_recast_array(registered, moving.dtype, remove_negative=True), (xoff, yoff))


def transform_2D_fast(moving: Union[np.ndarray, da.core.Array], parameters: Tuple) -> np.ndarray:
    """Transform an image using a defined parameters.

    Apply set of transformation parameters to an input
    image using the ``image_registration`` package.

    Parameters
    ----------
    moving
        an image to be transformed
    parameters
        (xoff, yoff) as returned by ``register_2D_fast``

    Returns
    -------
    numpy.ndarray
        transformed image derived by transforming ``moving``

    Raises
    ------
    TypeError
        If any of the positional inputs are not of the correct type
    ValueError
        If the input array is not rank 2
    """

    try:
        confirm_array_rank(moving, 2)
    except TypeError:
        raise TypeError("``moving`` is not of correct type (numpy.ndarray)")
    except ValueError:
        raise ValueError("``moving`` is not of correct rank (2)")

    if not isinstance(parameters, tuple):
        raise TypeError("``parameters is not a tuple``")
    if not all([isinstance(p, (float, int)) for p in parameters]):
        raise TypeError("``parameters elements are non-numeric``")

    # image_registration does not play nice with dask arrays,
    # compute these here to convert to numpy.ndarray
    if isinstance(moving, da.core.Array):
        moving = moving.compute()

    xoff, yoff = parameters
    # subpixel fitting using the following function
    # results in negative values and FFT artefacts
    # registered = reg.fft_tools.shift.shiftnd(moving, (-yoff, -xoff))
    xoff = int(np.round(xoff))
    yoff = int(np.round(yoff))
    registered = translate_array(moving, -yoff, -xoff)

    return _recast_array(registered, moving.dtype, remove_negative=True)


def _calculate_shifts_elastix(
    images: List[AICSImage],
    reference_channel: int = 0,
    reference_cycle: int = 0,
    settings: TransformationParameters = TransformationParameters("rigid"),
) -> List[TransformationParameters]:
    """Wrapper for register_2D."""

    fixed = images[reference_cycle].get_image_data("YX", C=reference_channel)
    transformation_parameters = [
        register_2D(
            fixed=fixed,
            moving=image.get_image_data("YX", C=reference_channel),
            settings=settings,
            parameters_only=True,
        )
        for image in images
    ]

    if not all([isinstance(p, TransformationParameters) for p in transformation_parameters]):
        raise TypeError("One or more of the transformation parameters calculated are not of the correct type")
    if len(transformation_parameters) != len(images):
        raise RuntimeError("Error calculating transformation parameters for one or more images")

    return transformation_parameters


def _calculate_shifts_image_registration(
    images: List[AICSImage], reference_channel: int = 0, reference_cycle: int = 0
) -> List[Tuple]:
    """Wrapper for register_2D_fast."""

    fixed = images[reference_cycle].get_image_data("YX", C=reference_channel)
    transformation_parameters = [
        register_2D_fast(
            fixed=fixed,
            moving=image.get_image_data("YX", C=reference_channel),
            parameters_only=True,
        )
        for image in images
    ]

    if not all([isinstance(p, tuple) for p in transformation_parameters]):
        raise TypeError("One or more of the transformation parameters calculated are not of the correct type")
    if len(transformation_parameters) != len(images):
        raise RuntimeError("Error calculating transformation parameters for one or more images")

    return transformation_parameters


@overload
def calculate_shifts(
    images: List[AICSImage],
    reference_channel: int,
    reference_cycle: int,
    lib: Literal["elastix"],
    registration_settings: Union[str, TransformationParameters, None],
) -> List[TransformationParameters]:
    ...


@overload
def calculate_shifts(
    images: List[AICSImage],
    reference_channel: int,
    reference_cycle: int,
    lib: Literal["image_registration"],
) -> List[Tuple]:
    ...


def calculate_shifts(
    images: List[AICSImage],
    reference_channel: int = 0,
    reference_cycle: int = 0,
    lib: str = "elastix",
    registration_settings: Union[str, TransformationParameters, None] = None,
) -> Union[List[TransformationParameters], List[Tuple]]:
    """
    Calculate the shift of each 2D image in a list of AICSImage objects relative to a reference image,
    using either the elastix or image_registration libraries.

    Parameters
    ----------
    images
        A list of AICSImage objects to be registered.
    reference_channel
        The channel index of the reference image. Default is 0.
    reference_cycle
        The cycle index of the reference image. Default is 0.
    lib
        The library to use for registration. Default is 'elastix'.
    registration_settings
        Either a string (e.g. 'translation', 'rigid' or 'affine'), or a
        TransformationParameters object, specifying the registration
        settings. Default is None, which results in 'rigid' transformation
        when lib='elastix'. See blimp.registration.TransformationParameters
        for more details. Ignored when lib='image_registration'

    Returns
    -------
    Union[List[TransformationParameters], List[Tuple]]
        elastix: a list of TransformationParameters objects
        image_registration: a list of tuples, representing the shift of each image.

    Raises
    ------
    ValueError
        If one or more of the AICSImage objects has non-uniform or incorrect dimensionality.
        If the alignment library is not recognised
    NotImplementedError
        If the images have a Z dimension greater than 1.
    """

    if not check_uniform_dimension_sizes(images, omit="C"):
        raise ValueError("Check input. One or more of the ``AICSImage``s has non-uniform or incorrect dimensionality")
    elif images[0].dims.Z > 1:
        raise NotImplementedError(
            "Images have shape {images[0].dims.shape}. However, "
            + "3D registration is not yet available through this interface."
        )
    elif len(images) == 1:
        logger.warn("Only one image in list. Registering a single image to itself.")

    if reference_channel < 0 or any([reference_channel >= img.dims.C for img in images]):
        raise IndexError(f"Channel {reference_channel} out of range for at least one image")
    if reference_cycle < 0:
        raise IndexError(f"Cycle {reference_cycle} < 0")
    if reference_cycle >= len(images):
        raise IndexError(f"Cycle {reference_cycle} exceeds the length of images ({len(images)})")

    if lib == "elastix":
        logger.info(f"Using ``elastix`` library to align {len(images)} x 2D images.")

        # define registration settings
        if isinstance(registration_settings, TransformationParameters):
            settings = registration_settings
        elif isinstance(registration_settings, str):
            settings = TransformationParameters(transformation_mode=registration_settings)
        else:
            logger.warn(
                """
                Using default 'rigid' transformation to align images.
                Specify alternative settings using the ``registration_settings`` argument.
                """
            )
            settings = TransformationParameters(transformation_mode="rigid")

        # return list of transformation parameters
        return _calculate_shifts_elastix(
            images=images, reference_channel=reference_channel, reference_cycle=reference_cycle, settings=settings
        )

    elif lib == "image_registration":
        logger.info(f"Using ``image_registration`` library to align {len(images)} x 2D images.")

        if registration_settings is not None:
            logger.warn(
                "Using ``image_registration`` library to align images "
                + "using translation. Ignoring ``registration_settings``"
            )

        # return list of transformation coordinates
        return _calculate_shifts_image_registration(
            images=images, reference_channel=reference_channel, reference_cycle=reference_cycle
        )

    else:
        raise ValueError(f"lib : {lib} is not recognised")


@overload
def apply_shifts(
    images: List[AICSImage],
    transformation_parameters: List[TransformationParameters],
    lib: Literal["elastix"],
) -> List[AICSImage]:
    ...


@overload
def apply_shifts(
    images: List[AICSImage],
    transformation_parameters: List[Tuple],
    lib: Literal["image_registration"],
) -> List[AICSImage]:
    ...


def apply_shifts(
    images: List[AICSImage],
    transformation_parameters: Union[List[TransformationParameters], List[Tuple]],
    lib: str = "elastix",
    crop: bool = False,
) -> List[AICSImage]:
    """
    Apply transformations to 2D images in a list of AICSImage objects,
    using either the elastix or image_registration libraries.

    Parameters
    ----------
    images
        A list of AICSImage objects to be registered.
    transformation_parameters
        A list of transformation parameters to be used.
        For lib='elastix', parameters should be a list of
        ``blimp.registration.TransformationParameters``. For
        lib='image_registration', parameters should be tuples of
        (y,x) shifts.
    lib
        The library to use for registration. Default is 'elastix'.
    registration_settings
        Either a string (e.g. 'translation', 'rigid' or 'affine'), or a
        TransformationParameters object, specifying the registration
        settings. Default is None, which results in 'rigid' transformation
        when lib='elastix'. See blimp.registration.TransformationParameters
        for more details. Ignored when lib='image_registration'
    crop
        Whether to crop the output image to the minimum rectangle that
        contains the input data (default = ``False``)

    Returns
    -------
    List[AICSImage]
        for (lib=='elastix'): a list of TransformationParameters objects
        for (lib=='image_registration'): a list of tuples, representing the shift of each image.

    Raises
    ------
    ValueError
        If one or more of the AICSImage objects has non-uniform or incorrect dimensionality.
    NotImplementedError
        If the images have a Z dimension greater than 1.
    """

    if not check_uniform_dimension_sizes(images, omit="C"):
        raise ValueError("Check input. One or more of the ``AICSImage``s has non-uniform or incorrect dimensionality")
    elif images[0].dims.Z > 1:
        raise NotImplementedError(
            "Images have shape {images[0].dims.shape}. However, "
            + "3D registration is not yet available through this interface."
        )
    elif len(images) == 1:
        logger.warn("Only one image in list. Registering a single image to itself.")

    if lib == "elastix":
        if not all([isinstance(p, TransformationParameters) for p in transformation_parameters]):
            raise ValueError(
                "Using lib='elastix' but not all transformation_parameters list elements are TransformationParameters"
            )

        registered_arrays = [
            np.stack(
                [
                    # transform_2D returns "YX" array,
                    # use expand_dims to add a "Z" dimension
                    np.expand_dims(
                        transform_2D(moving=image.get_image_dask_data("YX", C=channel), parameters=params),  # type: ignore
                        axis=0,
                    )
                    for channel in range(image.dims.C)
                ],
                axis=0,
            )
            for image, params in zip(images, transformation_parameters)
        ]

    elif lib == "image_registration":
        if not all([isinstance(p, tuple) for p in transformation_parameters]):
            raise ValueError(
                "Using lib='image_registration' but not all transformation_parameters list elements are tuple"
            )
        registered_arrays = [
            np.stack(
                [
                    # transform_2D returns "YX" array,
                    # use expand_dims to add a "Z" dimension
                    np.expand_dims(
                        transform_2D_fast(moving=image.get_image_dask_data("YX", C=channel), parameters=params),  # type: ignore
                        axis=0,
                    )
                    for channel in range(image.dims.C)
                ],
                axis=0,
            )
            for image, params in zip(images, transformation_parameters)
        ]
    else:
        raise NotImplementedError(f"lib : {lib} is not recognised")

    # copy metadata from originals
    registered_images = [
        AICSImage(registered, physical_pixel_sizes=original.physical_pixel_sizes, channel_names=original.channel_names)
        for registered, original in zip(registered_arrays, images)
    ]

    if crop:
        # get cropping mask
        if lib == "image_registration":
            mask = _get_cropping_mask_from_transformation_parameters(transformation_parameters, shape=(images[0].dims.Y, images[0].dims.X))  # type: ignore
        elif lib == "elastix":
            mask = _get_cropping_mask_from_transformation_parameters(transformation_parameters)  # type: ignore
        registered_images = [crop_image(reg_img, mask=mask) for reg_img in registered_images]

    return registered_images


@overload
def _get_cropping_mask_from_transformation_parameters(parameters: List[TransformationParameters]) -> np.ndarray:
    ...


@overload
def _get_cropping_mask_from_transformation_parameters(parameters: List[Tuple[int]], shape: Tuple[int]) -> np.ndarray:
    ...


def _get_cropping_mask_from_transformation_parameters(
    parameters: Union[List[TransformationParameters], List[Tuple[int]]], shape: Tuple[int] = None
) -> np.ndarray:
    """Calculate a boolean mask for cropping images after registration.

    The mask is calculated from a list of transformation parameters and
    represents the spatial region of a multi-channel image that is shared
    between all images transformed according to the transformation parameters.

    Parameters
    ----------
    parameters
        List of transformation parameters, in either the
        ``blimp.preprocessing.TransformationParameters`` class, or as a list
        of (y,x) tuples (see ``apply_shifts`` for more details)

    Returns
    -------
    numpy.ndarray
        Boolean array with True for include and False for exclude.

    Raises
    ------
    ValueError
        If not all parameters have the same size

    """
    if not isinstance(parameters, list):
        raise TypeError("``parameters`` must be a list")

    if all(isinstance(p, TransformationParameters) for p in parameters):
        logger.debug(f"Computing bounding box for list of {len(parameters)} elastix transformation parameters")

        # check that the size parameter matches between all inputs
        shapes = [
            (int(float(y)), int(float(x))) for x, y in [p.parameter_object.GetParameter(0, "Size") for p in parameters]  # type: ignore
        ]
        if any([s != shapes[0] for s in shapes]):
            raise ValueError("Not all ``TransformationParameters`` have the same size.")

        # initialise an array of ones
        ones = np.ones(shape=shapes[0], dtype=float)

        # transform 'ones' arrays to find the regions to crop
        masks = [transform_2D(ones, p) for p in parameters]  # type: ignore

        # find region with all ones
        if len(masks) > 1:
            mask = np.mean(masks, axis=0) == 1.0  # type: ignore
        else:
            mask = np.floor(masks[0]).astype(np.bool_)
    elif all(isinstance(p, tuple) for p in parameters):
        logger.debug(f"Computing bounding box for list of {len(parameters)} (x,y) coordinates")

        if shape is None:
            raise ValueError("When using a list of (y,x) shifts, ``shape`` must be specified.")

        # initialise an array of ones
        ones = np.ones(shape=shape, dtype=float)

        # transform 'ones' arrays to find the regions to crop
        masks = [transform_2D_fast(ones, p) for p in parameters]  # type: ignore

        # find region with all ones
        if len(masks) > 1:
            mask = np.mean(masks, axis=0) == 1.0  # type: ignore
        else:
            mask = np.floor(masks[0]).astype(np.bool_)
    else:
        raise TypeError(
            "``parameters`` should either be a list of " + "``TransformationParameters`` or a list of (y,x) shifts"
        )
    return mask


def _crop_array(arr: np.ndarray, mask: np.ndarray):
    rows, cols = np.where(mask)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    return arr[min_row : max_row + 1, min_col : max_col + 1]


def crop_image(image: AICSImage, mask: np.ndarray):
    old = image.get_image_data("TCZYX")
    new_yx = _crop_array(mask, mask).shape
    new = np.ndarray(shape=list(old.shape[:3]) + list(new_yx), dtype=image.dtype)
    for t in range(old.shape[0]):
        for c in range(old.shape[1]):
            for z in range(old.shape[2]):
                new[t, c, z, :, :] = _crop_array(arr=old[t, c, z, :, :], mask=mask)
    return AICSImage(new, physical_pixel_sizes=image.physical_pixel_sizes, channel_names=image.channel_names)
