from typing import List, Union, Literal

from basicpy import BaSiC as BasicIlluminationCorrection
from aicsimageio import AICSImage
from aicsimageio.transforms import transpose_to_dims
import numpy as np

from blimp.utils import (
    average_images,
    concatenate_images,
    check_uniform_dimension_sizes,
)


def fit_illumination_correction(
    reference_images: List[AICSImage], timelapse: bool = False, **kwargs
) -> List[BasicIlluminationCorrection]:

    try:
        if not check_uniform_dimension_sizes(reference_images):
            raise ValueError(
                "Check input. One or more of the ``reference_images`` has non-uniform or incorrect dimensionality"
            )
    except TypeError:
        raise TypeError("All reference images should be ``AICSImage``s")

    if not timelapse:
        # use the 'T' axis to concatenate images
        images = concatenate_images(reference_images, axis="T", order="append")
    else:
        # average each timepoint
        images = average_images(reference_images)

    illum_corr = [BasicIlluminationCorrection(**kwargs) for _ in range(images.dims.C)]
    for c in range(images.dims.C):
        illum_corr[c].fit(images.get_image_data("TYX", C=c))

    return illum_corr


def _correct_image(
    obj: List[BasicIlluminationCorrection],
    im: AICSImage,
    timelapse: Union[bool, Literal["multiplicative"], Literal["additive"]] = False,
):
    if im.dims.C != len(obj):
        raise ValueError(f"Incorrect number of correction objects, expected {im.dims.C}.")
    else:
        if timelapse is False:
            corr = np.stack(
                [
                    np.stack(
                        [
                            np.stack(
                                [
                                    obj[c].transform(
                                        im.get_image_data("YX", C=c, Z=z, T=t),
                                        timelapse=timelapse,
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
        else:
            # correct timelapse data in inner loop and reorder
            # dimensions afterwards
            corr_CZTYX = np.stack(
                [
                    np.stack(
                        [
                            obj[c].transform(
                                im.get_image_data("TYX", C=c, Z=z),
                                timelapse=timelapse,
                            )[0]
                            for z in range(im.dims.Z)
                        ],
                        axis=0,
                    )
                    for c in range(im.dims.C)
                ],
                axis=0,
            )
            corr = transpose_to_dims(corr_CZTYX, given_dims="CZTYX", return_dims="TCZYX")
    return AICSImage(corr, physical_pixel_sizes=im.physical_pixel_sizes, channel_names=im.channel_names)


def apply_illumination_correction(
    correction_objects: BasicIlluminationCorrection,
    images: Union[AICSImage, List[AICSImage]],
    timelapse: Union[bool, Literal["multiplicative"], Literal["additive"]] = False,
) -> Union[AICSImage, List[AICSImage]]:

    if isinstance(images, AICSImage):
        return _correct_image(correction_objects, images, timelapse=timelapse)
    elif isinstance(images, list):
        if not all(isinstance(image, AICSImage) for image in images):
            raise TypeError("Not all of ``images`` are of class AICSImage")
        return [_correct_image(correction_objects, image, timelapse=timelapse) for image in images]
    else:
        raise TypeError("``images`` should be an AICSImage or list of AICSImages")
