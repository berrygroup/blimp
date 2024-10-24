from typing import Union, Optional
from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import mahotas as mh

logger = logging.getLogger(__name__)


def segment_nuclei_cellpose(
    intensity_image: AICSImage,
    nuclei_channel: int = 0,
    model_type: str = "nuclei",
    pretrained_model: Union[str, Path, None] = None,
    diameter: Optional[int] = None,
    threshold: float = 0,
    flow_threshold: float = 0.4,
    normalize: Union[bool, dict] = True,
    timepoint: Union[int, None] = None,
) -> AICSImage:
    """Segment nuclei.

    Parameters
    ----------
    intensity_image
        intensity image (possibly 5D: "TCZYX")
    nuclei_channel
        channel number corresponding to nuclear stain
    threshold
        cellprob_threshold, float between [-6,+6] after which objects are discarded
    timepoint
        which timepoint should be segmented (optional,
        default None will segment all time-points)

    Returns
    -------
    AICSImage
        label_image
    """
    from cellpose import models

    if timepoint is None:
        nuclei_images = [
            intensity_image.get_image_data("ZYX", C=nuclei_channel, T=t) for t in range(intensity_image.dims[["T"]][0])
        ]
    else:
        nuclei_images = [intensity_image.get_image_data("ZYX", C=nuclei_channel, T=timepoint)]

    if pretrained_model is None:
        cellpose_model = models.CellposeModel(gpu=False, model_type=model_type)
        masks, flows, styles = cellpose_model.eval(
            nuclei_images,
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=threshold,
            normalize=normalize,
            do_3D=False,
        )
    else:
        cellpose_model = models.CellposeModel(gpu=False, pretrained_model=str(pretrained_model))
        masks, flows, styles = cellpose_model.eval(nuclei_images, channels=[0, 0])

    segmentation = AICSImage(
        np.stack(masks)[:, np.newaxis, np.newaxis, :],
        channel_names=["Nuclei"],
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    return segmentation


def expand_objects_watershed(
    seeds_image: np.ndarray, background_image: np.ndarray, intensity_image: np.ndarray
) -> np.ndarray:
    """Expand objects.

    Expands objects in `seeds_image` using a watershed transform
    on `intensity_image`.

    Parameters
    ----------
    seeds_image:
        objects that should be expanded
    background_image:
        regions in the image that should be considered background and should
        not be part of an object after expansion
    intensity_image:
        grayscale image; pixel intensities determine how far individual
        objects are expanded

    Returns
    -------
    numpy.ndarray
        expanded objects
    """
    # We compute the watershed transform using the seeds of the primary
    # objects and the additional seeds for the background regions. The
    # background regions will compete with the foreground regions and
    # thereby work as a stop criterion for expansion of primary objects.

    labels = np.where(seeds_image != 0, seeds_image, background_image)
    regions = mh.cwatershed(np.invert(intensity_image), labels)
    # Remove background regions
    n_objects = len(np.unique(seeds_image[seeds_image > 0]))
    regions[regions > n_objects] = 0

    # Ensure objects are separated
    lines = mh.labeled.borders(regions)
    regions[lines] = 0

    # Close holes in objects.
    foreground_mask = regions > 0
    holes = np.logical_xor(mh.close_holes(foreground_mask), foreground_mask)
    holes = mh.morph.dilate(holes)
    holes_labeled, n_holes = mh.label(holes)
    for i in range(1, n_holes + 1):
        fill_value = np.unique(regions[holes_labeled == i])[-1]
        fill_value = fill_value[fill_value > 0][0]
        regions[holes_labeled == i] = fill_value

    # Remove objects that are obviously too small, i.e. smaller than any of
    # the seeds (this could happen when we remove certain parts of objects
    # after the watershed region growing)
    primary_sizes = mh.labeled.labeled_size(seeds_image)
    if len(primary_sizes) > 1:
        min_size = np.min(primary_sizes[1:]) + 1
        regions = mh.labeled.filter_labeled(regions, min_size=min_size)[0]

    # Remove regions that don't overlap with seed objects and assign
    # correct labels to the other regions, i.e. those of the corresponding seeds.

    new_label_image, n_new_labels = mh.labeled.relabel(regions)
    lut = np.zeros(np.max(new_label_image) + 1, new_label_image.dtype)
    for i in range(1, n_new_labels + 1):
        orig_labels = seeds_image[new_label_image == i]
        orig_labels = orig_labels[orig_labels > 0]
        orig_count = np.bincount(orig_labels)
        orig_unique = np.where(orig_count)[0]
        if orig_unique.size == 1:
            lut[i] = orig_unique[0]
        elif orig_unique.size > 1:
            # logger.warn(
            #    'objects overlap after expansion: %s',
            #    ', '.join(map(str, orig_unique))
            # )
            lut[i] = np.where(orig_count == np.max(orig_count))[0][0]
    expanded_image = lut[new_label_image]

    # Ensure that seed objects are fully contained within expanded objects
    index = (seeds_image - expanded_image) > 0
    expanded_image[index] = seeds_image[index]

    return expanded_image


def segment_secondary(
    primary_label_image: np.ndarray,
    intensity_image: np.ndarray,
    contrast_threshold: float,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
) -> np.ndarray:
    """Segment Secondary.

    Detects secondary objects in an image by expanding the primary objects
    encoded in `primary_label_image`. The outlines of secondary objects are
    determined based on the watershed transform of `intensity_image` using the
    primary objects in `primary_label_image` as seeds.

    Parameters
    ----------
    primary_label_image: numpy.ndarray[numpy.int32]
        2D labeled array encoding primary objects, which serve as seeds for
        watershed transform
    intensity_image: numpy.ndarray[numpy.uint8 or numpy.uint16]
        2D grayscale array that serves as gradient for watershed transform;
        optimally this image is enhanced with a low-pass filter
    contrast_threshold: int
        contrast threshold for automatic separation of forground from background
        based on locally adaptive thresholding (when ``0`` threshold defaults
        to `min_threshold` manual thresholding)
    min_threshold: int, optional
        minimal foreground value; pixels below `min_threshold` are considered
        background
    max_threshold: int, optional
        maximal foreground value; pixels above `max_threshold` are considered
        foreground
    plot: bool, optional
        whether a plot should be generated

    Returns
    -------
    numpy.ndarray
        secondary_label_image

    Note
    ----
    Setting `min_threshold` and `max_threshold` to the same value reduces
    to manual thresholding.
    """
    if np.any(primary_label_image == 0):
        has_background = True
    else:
        has_background = False

    if not has_background:
        secondary_label_image = primary_label_image
    else:
        # We use adaptive thresholding to determine background regions,
        # i.e. regions in the intensity_image that should not be covered by
        # secondary objects.
        n_objects = len(np.unique(primary_label_image))
        # logger.info(
        #    'primary label image has %d objects',
        #    n_objects - 1
        # )
        if np.max(primary_label_image) != n_objects - 1:
            raise ValueError(f"Objects are not consecutively labeled, please relabel before secondary segmentation.")
        # SB: Added a catch for images with no primary objects
        # note that background is an 'object'
        if n_objects > 1:
            background_mask = mh.thresholding.bernsen(intensity_image, 5, contrast_threshold)
            if min_threshold is not None:
                # logger.info(
                #    'set lower threshold level to %d', min_threshold
                # )
                background_mask[intensity_image < min_threshold] = True

            if max_threshold is not None:
                # logger.info(
                #    'set upper threshold level to %d', max_threshold
                # )
                background_mask[intensity_image > max_threshold] = False
            background_label_image = (mh.label(background_mask)[0] > 0).astype(np.int32)
            if n_objects >= 2147483646:
                raise ValueError(f"Number of objects ({n_objects}) exceeds 32-bit datatype.")
            background_label_image[background_mask] += n_objects

            # logger.info('detect secondary objects via watershed transform')
            secondary_label_image = expand_objects_watershed(
                primary_label_image, background_label_image, intensity_image
            )
        else:
            # logger.info('skipping secondary segmentation')
            secondary_label_image = np.zeros(primary_label_image.shape, dtype=np.int32)

    n_objects = len(np.unique(secondary_label_image)[1:])
    # logger.info('identified %d objects', n_objects)

    return secondary_label_image
