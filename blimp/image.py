from typing import Any, Dict, Type, Optional
from pathlib import Path
import pickle
import logging

from aicsimageio import types, AICSImage, transforms
from aicsimageio.readers.reader import Reader
import numpy as np

from blimp.utils import translate_array
from blimp.preprocessing.registration import apply_shifts, TransformationParameters

logger = logging.getLogger(__name__)

# TODO: now that illumination statistics and transformation parameters,
# can be applied during image loading, it makes sense for these to be stored
# in a standard (configurable) location, and loaded from this location when required.

# TODO: update for new illuminationcorrection class


class BLImage(AICSImage):
    def __init__(
        self,
        image: types.ImageLike,
        reader: Optional[Type[Reader]] = None,
        reconstruct_mosaic: bool = True,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        super().__init__(image, reader, reconstruct_mosaic, fs_kwargs, **kwargs)
        self._illumination_correction_reference_image_files = None
        self._illumination_correction_objects = None
        self._illumination_correction_file = None
        self._transformation_parameters = None
        self._transformation_type = None
        self._transformation_parameters_file = None
        self._crop_mask_file = None
        self._crop_mask = None

    def __str__(self) -> str:
        return (
            f"<BLImage ["
            f"Reader: {type(self.reader).__name__}, "
            f"dtype: {self.dtype}, "
            f"Image-is-in-Memory: {self._xarray_data is not None}"
            f"]>"
        )

    def shift_yx(self, y: int, x: int):
        arr = np.stack(
            [
                np.stack(
                    [
                        np.stack(
                            [
                                translate_array(self.get_image_data("YX", Z=z, C=c, T=t), y, x)
                                for z in range(self.dims.Z)
                            ],
                            axis=0,
                        )
                        for c in range(self.dims.C)
                    ],
                    axis=0,
                )
                for t in range(self.dims.T)
            ],
            axis=0,
        )

        return self.__class__(arr, physical_pixel_sizes=self.physical_pixel_sizes, channel_names=self.channel_names)

    @property
    def illumination_correction_reference_image_files(self):
        return self._illumination_correction_reference_image_files

    @illumination_correction_reference_image_files.setter
    def illumination_correction_reference_image_files(self, paths):
        """Define which images should be used for correcting illumination.

        Parameters
        ----------
        paths
            a list of type str or pathlib.Path indicating the reference images

        Raises
        ------
        TypeError
            Paths provided are not of the correct type
        FileNotFoundError
            Path cannot be found
        RuntimeWarning
            One or more of the images is of incorrect shape or dimensionality
        """
        checked_reference_image_paths = []

        if isinstance(paths, list):
            # generate a list of paths
            reference_image_paths = []
            for path in paths:
                if isinstance(path, str):
                    reference_image_paths.append(Path(str))
                elif isinstance(path, Path):
                    reference_image_paths.append(path)
                else:
                    raise TypeError("Argument ``path`` must have type ``pathlib.Path`` or ``str``.")

            # check that these all exist and have the same type and dimensionality
            for path in reference_image_paths:
                if not path.exists():
                    raise FileNotFoundError(f"Illumination correction file at {path} does not exist.")
                image = AICSImage(path)
                if image.dims == self.dims and image.dtype == self.dtype:
                    checked_reference_image_paths.append(path)
                else:
                    raise RuntimeWarning(
                        f"Reference image has dimension {image.dims} and type {image.dtype},"
                        + f" expected {self.dims} ({self.dtype}). Omitting {path} from reference images."
                    )
            self._illumination_correction_reference_image_files = checked_reference_image_paths
        elif paths is None:
            self._illumination_correction_reference_image_files = None
        else:
            raise TypeError(
                "Argument ``illumination_correction_reference_image_files`` should be a list of type ``str`` or ``pathlib.Path``."
            )

    @property
    def illumination_correction_objects(self):
        """List of length BLImage.dims.C, containing objects to correct illumination in each channel."""
        return self._illumination_correction_objects

    @property
    def illumination_correction_file(self):
        """The path where illumination_correction_objects is saved."""
        return self._illumination_correction_file

    def _load_illumination_correction_objects(self):
        if self._illumination_correction_objects is None:
            if self._illumination_correction_file.exists():
                obj_list = pickle.load(self._illumination_correction_file)
                if len(obj_list) == self.dims.C:
                    self._illumination_correction_objects = obj_list
                else:
                    raise RuntimeError(
                        "Illumination correction object loaded from "
                        + f"{str(self._illumination_correction_file)} has "
                        + "incorrect number of channels."
                    )
            else:
                raise RuntimeError(
                    "Illumination correction object not found. Have you "
                    + "called the ``fit_illumination_correction()`` method?"
                )

    # TODO: update for new objet oriented illum corr
    #    def fit_illumination_correction(self, timelapse: bool = False, **kwargs) -> None:
    #        if self._illumination_correction_reference_image_files is None:
    #            raise RuntimeError(
    #                "Before fitting, reference images for illumination correction must be defined."
    #                + "Try ``BLImage.illumination_correction_reference_image_files = [paths/to/files]``"
    #            )
    #        else:
    #            reference_images = [AICSImage(f) for f in self._illumination_correction_reference_image_files]
    #            self._illumination_correction_objects = fit_illumination_correction(
    #                reference_images=reference_images, timelapse=timelapse, **kwargs
    #            )
    #            if Path(self._illumination_correction_file).exists():
    #                logger.warning(
    #                    f"Illumination correction file {self._illumination_correction_file} already exists. Replacing with new file."
    #                )
    #            else:
    #                logger.debug(f"Writing illumination correction file at {self._illumination_correction_file}.")
    #            pickle.dump(obj=self._illumination_correction_objects, file=self._illumination_correction_file)

    @property
    def transformation_type(self):
        """The type of transformation parameters being used."""
        return self._transformation_type

    @property
    def transformation_parameters_file(self):
        """The path where transformation_parameters is saved."""
        return self._transformation_parameters_file

    @property
    def transformation_parameters(self):
        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, parameters):
        if isinstance(parameters, TransformationParameters):
            self._transformation_type = "elastix"
        elif isinstance(parameters, tuple):
            self._transformation_type = "image_registration"
        else:
            raise TypeError(
                "Argument ``parameters`` must have type ``blimp.preprocessing.registration.TransformationParameters`` or ``tuple``."
            )
        self._transformation_parameters = parameters

    @property
    def crop_mask(self):
        return self._crop_mask

    @property
    def crop_mask_file(self):
        return self._crop_mask_file

    #    def define_crop_mask(self):
    #
    #    def _load_crop_mask(self):
    #        if self._crop_mask is None:
    #            if self._crop_mask_file.exists():
    #                self._crop_mask = np.load(self._crop_mask_file)
    #            else:
    #                raise RuntimeError(
    #                    "Illumination correction object not found. Have you called the ``define_crop_mask()`` method?"
    #                )

    def get_image_data(
        self,
        dimension_order_out: Optional[str] = None,
        align: bool = False,
        crop: bool = False,
        correct: bool = False,
        timelapse=None,
        **kwargs: Any,
    ) -> np.ndarray:
        # first get a standardised representation as a numpy array
        arr = super().get_image_data("TCZYX", **kwargs)
        if timelapse is None:
            timelapse = self.dims.T > 1

        # 1. Illumination correction
        if correct is True:
            self._load_illumination_correction_objects()
        # TODO: update
        #            arr = correct_illumination(
        #                images=arr, correction_objects=self._illumination_correction_objects, timelapse=timelapse
        #            )

        # 2. Alignment
        if align is True:
            if self._transformation_parameters is None:
                raise RuntimeError("Transformation parameters not defined")
            else:
                arr = apply_shifts(
                    images=AICSImage(arr),
                    transformation_parameters=self._transformation_parameters,
                    lib=self._transformation_type,
                ).get_image_data("TCZYX")

        return transforms.reshape_data(
            data=arr,
            given_dims="TCZYX",
            return_dims=dimension_order_out,
            **kwargs,
        )


#    def get_image_dask_data(
#        self,
#        dimension_order_out: Optional[str] = None,
#        **kwargs: Any) -> Union[da.Array, np.ndarray]:
#        dask_arr = super().get_image_dask_data(dimension_order_out, **kwargs)
#        if align is True:
#            logger.warn("Cannot return dask.array.Array with align True, returning numpy.array")
#            arr = transform_2D(arr)
#        if correct_illumination is True:
#            logger.warn("Cannot return dask.array.Array with correct_illumination True, returning numpy.array")
#            arr = correct_illumination(arr)
#        return arr

#        return dask_arr
