# pip install itk-elastix
from typing import List, Tuple, Union
from pathlib import Path
import logging

from itk import ParameterObject, transformix_filter, elastix_registration_method
import numpy as np

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


def check_arrays_are_2D(images: Union[np.ndarray, List[np.ndarray]]) -> None:
    if isinstance(images, np.ndarray):
        if len(images.shape) != 2:
            raise ValueError("Input is not a 2D array (shape = {images.shape})")

    elif isinstance(images, list):
        for image in images:
            if not isinstance(image, np.ndarray):
                raise TypeError("Input should be of type np.ndarray or list of np.ndarray")
            elif len(image.shape) != 2:
                raise ValueError("Input is not a 2D array (shape = {images.shape})")
    else:
        raise TypeError("Input should be of type numpy.ndarray or list of numpy.ndarray")
    return None


def register_2D(
    fixed: np.ndarray, moving: np.ndarray, settings: TransformationParameters
) -> Tuple[np.ndarray, TransformationParameters]:
    """Align an image to a reference image, keeping transformation parameters

    A ``moving`` image is aligned to a ``fixed`` image using
    ITK Elastix.

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
        check_arrays_are_2D([fixed, moving])
    except TypeError:
        # logger.error("Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray)")
        pass
    except ValueError:
        # logger.error("Either ``fixed`` or ``moving`` are not of correct rank (2)")
        pass

    registered, parameters = elastix_registration_method(
        np.asarray(fixed, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        parameter_object=settings.parameter_object,
    )

    return (np.asarray(registered, dtype=fixed.dtype), TransformationParameters(from_object=parameters))


def transform_2D(moving: np.ndarray, parameters: TransformationParameters) -> np.ndarray:
    """Transform an image using a defined parameters

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
        check_arrays_are_2D(moving)
    except TypeError:
        # logger.error("Either ``fixed`` or ``moving`` are not of correct type (numpy.ndarray)")
        pass
    except ValueError:
        # logger.error("Either ``fixed`` or ``moving`` are not of correct rank (2)")
        pass

    transformed = transformix_filter(np.asarray(moving, dtype=np.float32), parameters.parameter_object)

    return np.asarray(transformed, dtype=moving.dtype)

    def register_2D_fast_translation(fixed, moving):
        import image_registration as reg

        # TODO finish off this approach using the image_registration package
        xoff, yoff, exoff, eyoff = reg.chi2_shift(fixed, moving)
        registered = reg.fft_tools.shift.shiftnd(moving, (-yoff, -xoff))
        return (registered, (xoff, yoff))
