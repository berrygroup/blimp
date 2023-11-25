from typing import List, Union, Literal, Optional
from pathlib import Path
import pickle
import logging

from matplotlib import pyplot as plt
from aicsimageio import readers, AICSImage
from aicsimageio.transforms import reshape_data
import numpy as np
import basicpy
import dask.array as da

from blimp.utils import (
    equal_dims,
    average_images,
    mean_std_welford,
    concatenate_images,
    convert_array_dtype,
    check_uniform_dimension_sizes,
)

logger = logging.getLogger(__name__)


class IlluminationCorrection:
    def __init__(
        self,
        method: Union[Literal["pixel_z_score"], Literal["basic"]] = "pixel_z_score",
        reference_images: Union[List[AICSImage], List[str], List[Path], None] = None,
        timelapse: Union[bool, Literal["multiplicative"], Literal["additive"], None] = None,
        from_file: Union[str, Path, None] = None,
        **kwargs,
    ):
        self._method = method
        self._timelapse = timelapse
        self._dims = None
        self._file_path = None
        self._file_name = None
        self._correctors = None
        self._mean_image = None
        self._std_image = None
        self._mean_mean_image = None
        self._mean_std_image = None

        # 1. Initialise using reference images and set correctors using fit()
        if reference_images is not None:
            logger.debug("Initialising ``IlluminationCorrection`` using ``reference_images``.")
            if not isinstance(reference_images, list):
                raise TypeError("``reference images`` must be a list")
            else:
                # check ``reference_images`` input lists
                is_input_AICSImage = all(isinstance(image, AICSImage) for image in reference_images)
                is_input_files = all(isinstance(image, (str, Path)) for image in reference_images)
                if (not is_input_AICSImage) and (not is_input_files):
                    raise TypeError(
                        "``reference images`` must be a list of ``AICSImage``s or file paths "
                        + "(``str`` or ``pathlib.Path``)"
                    )
                if is_input_files and not all(Path(image).is_file() for image in reference_images):
                    raise FileNotFoundError

                # check timelapse defined
                if self._timelapse is None:
                    raise ValueError(
                        "``timelapse`` must be specified when initialising "
                        + "``IlluminationCorrection`` using ``reference_images``"
                    )

                # call the fit method to initialise self._correctors
                # pass on **kwargs to allow basicpy.BaSiC optimisation
                if is_input_AICSImage:
                    self.fit(reference_images, **kwargs)  # type: ignore
                elif is_input_files:
                    images = [
                        AICSImage(image, reader=readers.ome_tiff_reader.OmeTiffReader) for image in reference_images
                    ]
                    self.fit(images, **kwargs)

        # 2. Read from a file using load()
        elif from_file is not None:
            logger.debug(f"Loading ``IlluminationCorrection`` from {from_file}.")
            self.file_path = from_file
            self.load(self._file_path)  # type: ignore
            if self._timelapse is None:
                raise ValueError("``timelapse`` is not specified in file {self._file_path}")
            if self._method is None:
                raise ValueError("``method`` is not specified in file {self._file_path}")
            elif self._method == "pixel_z_score":
                # FIXME: recompute mean_mean and mean_std on loading as these seem to not be stored in file
                self._mean_mean_image = [
                    np.mean(self._mean_image.get_image_data("YX", C=c)) for c in range(self._dims.C)
                ]
                self._mean_std_image = [np.mean(self._std_image.get_image_data("YX", C=c)) for c in range(self._dims.C)]

        # 3. Initialise empty
        else:
            logger.debug("Initialising empty ``IlluminationCorrection`` object.")

    @property
    def dims(self):
        return self._dims

    @property
    def method(self):
        return self._method

    @property
    def timelapse(self):
        return self._timelapse

    @property
    def mean_image(self):
        return self._mean_image

    @property
    def std_image(self):
        return self._std_image

    @property
    def mean_mean_image(self):
        return self._mean_mean_image

    @property
    def mean_std_image(self):
        return self._mean_std_image

    @property
    def correctors(self):
        return self._correctors

    def fit(self, reference_images: List[AICSImage], timelapse: bool = False, **kwargs):
        try:
            if not check_uniform_dimension_sizes(reference_images):
                raise ValueError(
                    "Check input. One or more of the ``reference_images`` has non-uniform or incorrect dimensionality"
                )
        except TypeError:
            raise TypeError("All reference images should be ``AICSImage``s")

        self._dims = reference_images[0].dims

        if self._method == "pixel_z_score":
            # Use the Welford method, which computes mean and std using data
            # loaded in series (to reduce the memory requirement)
            if not self._timelapse:
                self._mean_image, self._std_image = mean_std_welford(reference_images)
                self._mean_mean_image = [
                    np.mean(self._mean_image.get_image_data("YX", C=c)) for c in range(self._dims.C)
                ]
                self._mean_std_image = [np.mean(self._std_image.get_image_data("YX", C=c)) for c in range(self._dims.C)]
            else:
                raise NotImplementedError(
                    "``pixel_z_score`` method not implemented for timelapse data. "
                    + "Set ``timelapse=False`` to calculate a constant correction across time"
                )
        elif self._method == "basic":
            # Use pybasic
            if not self._timelapse:
                # use the 'T' axis to concatenate images
                images = concatenate_images(reference_images, axis="T", order="append")
            else:
                # if timelapse, average each timepoint
                images = average_images(reference_images)

            self._correctors = [basicpy.BaSiC(**kwargs) for _ in range(self._dims.C)]
            for c in range(images.dims.C):
                self._correctors[c].fit(images.get_image_data("TYX", C=c))

    def plot(self):
        if self._method == "basic":
            if isinstance(self._correctors, list):
                fig, axes = plt.subplots(self.dims.C, 3, figsize=(9, 3 * self.dims.C), squeeze=False)
                for i in range(self.dims.C):
                    im = axes[i, 0].imshow(self.correctors[i].flatfield)
                    fig.colorbar(im, ax=axes[i, 0])
                    axes[i, 0].set_title("Flatfield")
                    im = axes[i, 1].imshow(self.correctors[i].darkfield)
                    fig.colorbar(im, ax=axes[i, 1])
                    axes[i, 1].set_title("Darkfield")
                    axes[i, 2].plot(self.correctors[i].baseline)
                    axes[i, 2].set_xlabel("Frame")
                    axes[i, 2].set_ylabel("Baseline")
                fig.tight_layout()
            else:
                raise RuntimeError("Cannot plot ``IlluminationCorrection`` if correctors are not defined")

        elif self._method == "pixel_z_score":
            if isinstance(self._mean_image, AICSImage):
                fig, axes = plt.subplots(self.dims.C, 2, figsize=(9, 3 * self.dims.C), squeeze=False)
                for i in range(self.dims.C):
                    im_dat = self.mean_image.get_image_data("YX", C=i)
                    upp = np.quantile(im_dat, 0.95)
                    im = axes[i, 0].imshow(im_dat, vmin=0, vmax=upp)
                    fig.colorbar(im, ax=axes[i, 0])
                    axes[i, 0].set_title("Mean image")

                    im_dat = self.std_image.get_image_data("YX", C=i)
                    upp = np.quantile(im_dat, 0.95)
                    im = axes[i, 1].imshow(im_dat, vmin=0, vmax=upp)
                    fig.colorbar(im, ax=axes[i, 1])
                    axes[i, 1].set_title("Std. image")

                fig.tight_layout()
            else:
                raise RuntimeError("``mean_image`` not defined, cannot plot ``IlluminationCorrection``")

    @property
    def file_name(self):
        return self._file_name

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: Union[str, Path]):
        if not isinstance(file_path, (str, Path)):
            raise TypeError("``file_path`` must be a ``str`` or ``pathlib.Path``.")

        p = Path(file_path)
        if p.suffix == "":
            raise ValueError("``path`` must be a file, not a directory.")
        if not p.parent.exists():
            logger.debug("Creating folder for illumination correction files")
            p.parent.mkdir()

        self._file_path = p
        self._file_name = self._file_path.name

    def save(self, path: Union[str, Path]):
        self.file_path = path
        if self._method == "basic":
            if isinstance(self._correctors, list):
                with open(self._file_path, "wb") as f:  # type: ignore
                    pickle.dump(self, f)
            else:
                raise RuntimeError(
                    "Cannot save ``IlluminationCorrection`` if correctors are not defined (method = ``basic``)"
                )
        elif self._method == "pixel_z_score":
            if (
                isinstance(self._mean_image, AICSImage)
                and isinstance(self._std_image, AICSImage)
                and isinstance(self._file_path, Path)
            ):
                with open(self._file_path, "wb") as f:  # type: ignore
                    pickle.dump(self, f)
                # also save mean_image and std_image in the same directory for ease of external validation
                self._mean_image.save(self._file_path.parent / (self._file_path.stem + "_mean_image.ome.tiff"))
                self._std_image.save(self._file_path.parent / (self._file_path.stem + "_std_image.ome.tiff"))
            else:
                raise RuntimeError(
                    "Cannot save ``IlluminationCorrection`` if mean and std have not been calculated (method = ``basic``)"
                )

    def load(self, path: Union[str, Path, None] = None):
        # 1. Check inputs
        if isinstance(path, (str, Path)):
            path = Path(path)
        elif path is None:
            if self._file_path is None:
                raise ValueError("``file_path`` attribute must be set, or passed to ``load`` method")
            else:
                path = self._file_path
        else:
            raise TypeError

        # 2. Load from file
        if not path.is_file():  # type: ignore
            raise FileNotFoundError(f"{str(path)} not found")
        with open(path, "rb") as f:  # type: ignore
            illumination_correction = pickle.load(f)

        # 3. Check attributes of loaded object
        if str(illumination_correction._file_path) != str(path):
            logger.warn(
                f"``file_path =`` {str(illumination_correction._file_path)} in object "
                + f"stored at {str(path)} does not match. Overwriting ``file_path`` attribute"
            )
            illumination_correction.file_path = path
        if illumination_correction._dims is None:
            raise RuntimeError(f"Object at {path} has no ``dims`` attribute.")
        if illumination_correction._timelapse is None:
            raise RuntimeError(f"Object at {path} has no ``timelapse`` attribute.")
        if illumination_correction._method is None:
            raise RuntimeError(f"Object at {path} has no ``method`` attribute.")

        if illumination_correction._method == "basic":
            if not isinstance(illumination_correction._correctors, list):
                raise RuntimeError(f"Object at {path} does not have a list of correctors.")
            if not all(isinstance(c, basicpy.BaSiC) for c in illumination_correction._correctors):
                raise RuntimeError(f"Correctors in object at {path} have unrecognised type.")
            if illumination_correction._dims.C != len(illumination_correction._correctors):
                raise RuntimeError(
                    f"Object at {path} has ``dims`` = {illumination_correction._dims.C} "
                    + f"but only {len(illumination_correction._correctors)} correctors."
                )
        elif illumination_correction._method == "pixel_z_score":
            pass
        else:
            raise RuntimeError(f"Object at {path} has unrecognised ``method`` attribute.")

        # 4. Set attributes
        self._dims = illumination_correction.dims
        self._timelapse = illumination_correction.timelapse

        if illumination_correction._method == "basic":
            self._correctors = illumination_correction.correctors
        elif illumination_correction._method == "pixel_z_score":
            self._mean_image = illumination_correction.mean_image
            self._std_image = illumination_correction.std_image

    def correct(
        self, image: Union[AICSImage, np.ndarray, da.core.Array, List[AICSImage], List[np.ndarray], List[da.core.Array]]
    ):
        return correct_illumination(image, self)


def pixel_z_score(
    original: np.ndarray, mean_image: np.ndarray, std_image: np.ndarray, mean_mean_image: float, mean_std_image: float
) -> np.ndarray:
    z = (original.astype(float) - mean_image) / std_image
    corrected = mean_mean_image + (mean_std_image * z)

    if original.dtype.kind in ["i", "u"]:
        corrected = np.rint(corrected).astype(original.dtype)
    else:
        corrected = corrected.astype(original.dtype)

    return corrected


def _correct_illumination(
    image: Union[AICSImage, np.ndarray, da.core.Array],
    illumination_correction: IlluminationCorrection,
    dimension_order_in: Optional[str] = None,
) -> Union[AICSImage, np.ndarray]:
    # 1. check input types and convert to AICSImage
    if isinstance(image, np.ndarray):
        if dimension_order_in is None:
            raise ValueError("``dimension_order_in`` must be specified for array inputs to ``_correct_illumination``")
        im = AICSImage(reshape_data(data=image, given_dims=dimension_order_in, return_dims="TCZYX"))
        out_type = "ndarray"
    elif isinstance(image, da.core.Array):
        if dimension_order_in is None:
            raise ValueError("``dimension_order_in`` must be specified for array inputs to ``_correct_illumination``")
        im = AICSImage(reshape_data(data=image, given_dims=dimension_order_in, return_dims="TCZYX"))
        out_type = "ndarray"
    elif isinstance(image, AICSImage):
        im = image
        out_type = "AICSImage"
    else:
        out_type = None
        raise TypeError(f"Unknown input image type {type(image)}")

    # 2a. correct data where the same correction is applied to all time-points
    if illumination_correction.timelapse is False:
        if not equal_dims(im, illumination_correction, dimensions="CYX"):
            raise ValueError(
                "``IlluminationCorrection`` object has incorrect ``dims``: expected "
                + f"{im.dims}, found {illumination_correction.dims}."
            )
        if illumination_correction.method == "basic":
            corr = np.stack(
                [
                    np.stack(
                        [
                            np.stack(
                                [
                                    illumination_correction.correctors[c].transform(
                                        im.get_image_data("YX", C=c, Z=z, T=t),
                                        timelapse=illumination_correction.timelapse,
                                    )[0]
                                    for z in range(im.dims.Z)
                                ],
                                axis=0,
                            )
                            for c in range(im.dims.C)
                        ],
                        axis=0,
                    )
                    for t in range(im.dims.T)
                ],
                axis=0,
            )
        elif illumination_correction.method == "pixel_z_score":
            corr = np.stack(
                [
                    np.stack(
                        [
                            np.stack(
                                [
                                    pixel_z_score(
                                        original=im.get_image_data("YX", C=c, Z=z, T=t),
                                        mean_image=illumination_correction.mean_image.get_image_data(
                                            "YX", C=c, Z=z, T=0
                                        ),
                                        std_image=illumination_correction.std_image.get_image_data("YX", C=c, Z=z, T=0),
                                        mean_mean_image=illumination_correction.mean_mean_image[c],
                                        mean_std_image=illumination_correction.mean_std_image[c],
                                    )
                                    for z in range(im.dims.Z)
                                ],
                                axis=0,
                            )
                            for c in range(im.dims.C)
                        ],
                        axis=0,
                    )
                    for t in range(im.dims.T)
                ],
                axis=0,
            )
    # 2b. correct data where corrections are time-dependent
    else:
        if illumination_correction.method == "pixel_z_score":
            raise NotImplementedError(
                "``pixel_z_score`` method not implemented for timelapse data. "
                + "Set ``timelapse=False`` to calculate a constant correction across time"
            )
        if not equal_dims(im, illumination_correction, dimensions="TCYX"):
            raise ValueError(
                "``IlluminationCorrection`` object has incorrect ``dims``: expected "
                + f"{im.dims}, found {illumination_correction.dims}."
            )
        # correct timelapse data in inner loop and reorder
        # dimensions afterwards
        corr_CZTYX = np.stack(
            [
                np.stack(
                    [
                        illumination_correction.correctors[c].transform(
                            im.get_image_data("TYX", C=c, Z=z),
                            timelapse=illumination_correction.timelapse,
                        )[0]
                        for z in range(im.dims.Z)
                    ],
                    axis=0,
                )
                for c in range(im.dims.C)
            ],
            axis=0,
        )
        corr = reshape_data(corr_CZTYX, given_dims="CZTYX", return_dims="TCZYX")

    # check dtype has not changed during correction
    if corr.dtype != im.dtype:
        corr = convert_array_dtype(corr, im.dtype, round_floats_if_necessary=True, copy=False)
    if out_type == "AICSImage":
        return AICSImage(corr, physical_pixel_sizes=im.physical_pixel_sizes, channel_names=im.channel_names)
    else:
        return reshape_data(data=corr, given_dims="TCZYX", return_dims=dimension_order_in)


def correct_illumination(
    images: Union[AICSImage, np.ndarray, da.core.Array, List[AICSImage], List[np.ndarray], List[da.core.Array]],
    illumination_correction: IlluminationCorrection,
    dimension_order_in: str = "TCZYX",
) -> Union[AICSImage, np.ndarray, List[AICSImage], List[np.ndarray]]:
    # individual input -> individual output
    if isinstance(images, (AICSImage, np.ndarray, da.core.Array)):
        res = _correct_illumination(images, illumination_correction, dimension_order_in)
    # list input -> list output
    elif isinstance(images, list):
        if all(isinstance(image, AICSImage) for image in images):
            res = [_correct_illumination(image, illumination_correction) for image in images]
        elif all(isinstance(image, np.ndarray) for image in images):
            res = [_correct_illumination(image, illumination_correction, dimension_order_in) for image in images]
        elif all(isinstance(image, da.core.Array) for image in images):
            res = [_correct_illumination(image, illumination_correction, dimension_order_in) for image in images]
    else:
        raise TypeError("``images`` is a non-uniform list")

    return res
