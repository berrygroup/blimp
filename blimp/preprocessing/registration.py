# pip install itk-elastix
from typing import List, Tuple, Union, Literal, overload
from pathlib import Path
import sys
import logging

from itk import ParameterObject, transformix_filter, elastix_registration_method
from aicsimageio import AICSImage
import numpy as np
import dask.array as da

from blimp.utils import confirm_array_rank, check_uniform_dimension_sizes

logger = logging.getLogger(__name__)


class TransformationParameters:
    def __init__(
        self,
        transformation_mode: Union[str, None] = None,
        resolutions: int = 3,
        from_object: Union[ParameterObject, None] = None,
        from_file: Union[str, None] = None,
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
            self.parameter_object = ParameterObject.New()

        else:
            logger.debug(f"Initialising ``TransformationParameters`` using defaults: {transformation_mode}")
            self.parameter_object = ParameterObject.New()
            if transformation_mode in ["translation", "rigid", "affine"]:
                parameter_map = self.parameter_object.GetDefaultParameterMap(transformation_mode, resolutions)
            else:
                logger.error("Unknown transformation mode {transformation_mode}")
                ValueError(f"ITK transformation_mode {transformation_mode} unknown")

            self.parameter_object.AddParameterMap(parameter_map)

    def save(self, file_name: Union[str, Path]):
        logger.debug("Saving transformation parameters as {file_name}")
        ParameterObject.WriteParameterFile(self.parameter_object, str(file_name))


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
    """

    try:
        confirm_array_rank([fixed, moving], 2)
    except TypeError:
        logger.error("Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray)")
        sys.exit()
    except ValueError:
        logger.error("Either ``fixed`` or ``moving`` are not of correct rank (2)")
        sys.exit()

    registered, parameters = elastix_registration_method(
        np.asarray(fixed, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        parameter_object=settings.parameter_object,
    )
    if parameters_only:
        return TransformationParameters(from_object=parameters)
    else:
        return (np.asarray(registered, dtype=fixed.dtype), TransformationParameters(from_object=parameters))


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
    """

    try:
        confirm_array_rank(moving, 2)
    except TypeError:
        logger.error("Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray)")
        sys.exit()
    except ValueError:
        logger.error("Either ``fixed`` or ``moving`` are not of correct rank (2)")
        sys.exit()

    transformed = transformix_filter(np.asarray(moving, dtype=np.float32), parameters.parameter_object)

    return np.asarray(transformed, dtype=moving.dtype)


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
    """
    import image_registration as reg

    xoff, yoff, exoff, eyoff = reg.chi2_shift(fixed, moving)
    registered = reg.fft_tools.shift.shiftnd(moving, (-yoff, -xoff))

    if parameters_only:
        return (xoff, yoff)
    else:
        return (registered, (xoff, yoff))


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
    """
    import image_registration as reg

    xoff, yoff = parameters
    registered = reg.fft_tools.shift.shiftnd(moving, (-yoff, -xoff))

    return registered


def _calculate_shifts_elastix(
    images: List[AICSImage],
    reference_channel: int = 0,
    reference_cycle: int = 0,
    settings: TransformationParameters = TransformationParameters("rigid"),
) -> List[TransformationParameters]:
    """Wrapper for register_2D."""

    fixed = images[reference_cycle].get_image_dask_data("YX", C=reference_cycle)
    transformation_parameters = []
    # TODO: use the new ``parameters_only`` to reformat into a list comprehension
    for image in images:
        # do not store images
        image_reg, params = register_2D(
            fixed=fixed, moving=image.get_image_dask_data("YX", C=reference_cycle), settings=settings
        )
        transformation_parameters.append(params)

    if not all([isinstance(p, TransformationParameters) for p in transformation_parameters]):
        raise TypeError("One or more of the transformation parameters calculated are not of the correct type")
    if len(transformation_parameters) != len(images):
        raise RuntimeError("Error calculating transformation parameters for one or more images")

    return transformation_parameters


def _calculate_shifts_image_registration(
    images: List[AICSImage], reference_channel: int = 0, reference_cycle: int = 0
) -> List[Tuple]:
    """Wrapper for register_2D_fast."""

    fixed = images[reference_cycle].get_image_dask_data("YX", C=reference_cycle)
    transformation_parameters = []
    # TODO: use the new ``parameters_only`` to reformat into a list comprehension
    for image in images:
        # do not store images
        image_reg, params = register_2D_fast(fixed=fixed, moving=image.get_image_dask_data("YX", C=reference_cycle))
        transformation_parameters.append(params)

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
    registration_settings: Union[str, TransformationParameters, None],
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
        The library to use for registration. Default is "elastix".
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
    NotImplementedError
        If the images have a Z dimension greater than 1.
    """

    if not check_uniform_dimension_sizes(images):
        raise ValueError("Check input. One or more of the ``AICSImage``s has non-uniform or incorrect dimensionality")
    elif images[0].dims.Z > 1:
        raise NotImplementedError(
            "Images have shape {images[0].dims.shape}. However, 3D registration is not yet available through this interface."
        )
    elif len(images) == 1:
        logger.warn("Only one image in list. Registering a single image to itself.")

    if lib == "elastix":
        logger.info(f"Using ``elastix`` library to align {len(images)} x 2D images.")

        # define registration settings
        if isinstance(registration_settings, TransformationParameters):
            settings = registration_settings
        elif isinstance(registration_settings, str):
            settings = TransformationParameters(transformation_mode=registration_settings)
        else:
            logger.warn("Using default 'rigid' transformation to align images.")
            settings = TransformationParameters(transformation_mode=registration_settings)

        # return list of transformation parameters
        return _calculate_shifts_elastix(
            images=images, reference_channel=reference_channel, reference_cycle=reference_cycle, settings=settings
        )

    elif lib == "image_registration":
        logger.info(f"Using ``image_registration`` library to align {len(images)} x 2D images.")

        if registration_settings is not None:
            logger.warn(
                "Using ``image_registration`` library to align images using translation. Ignoring ``registration_settings``"
            )

        # return list of transformation coordinates
        return _calculate_shifts_image_registration(
            images=images, reference_channel=reference_channel, reference_cycle=reference_cycle
        )

    else:
        raise NotImplementedError(f"lib : {lib} is not recognised")


def apply_shifts(
    images: List[AICSImage],
    transformation_parameters: Union[List[TransformationParameters], List[Tuple]],
    lib: str = "elastix",
) -> List[AICSImage]:

    if not check_uniform_dimension_sizes(images):
        raise ValueError("Check input. One or more of the ``AICSImage``s has non-uniform or incorrect dimensionality")
    elif images[0].dims.Z > 1:
        raise NotImplementedError(
            "Images have shape {images[0].dims.shape}. However, 3D registration is not yet available through this interface."
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

    return registered_images
