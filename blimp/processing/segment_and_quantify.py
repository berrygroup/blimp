from typing import List, Union, Optional, Tuple
from pathlib import Path
from functools import reduce
import logging

from aicsimageio import AICSImage
from skimage.segmentation import clear_border
import numpy as np
import pandas as pd
import mahotas as mh
import skimage.measure

HARALICK_BASE_NAMES = [
    "angular-second-moment",
    "contrast",
    "correlation",
    "sum-of-squares",
    "inverse-diff-moment",
    "sum-avg",
    "sum-var",
    "sum-entropy",
    "entropy",
    "diff-var",
    "diff-entropy",
    "info-measure-corr-1",
    "info-measure-corr-2",
]


def segment_nuclei_cellpose(
    intensity_image: AICSImage,
    nuclei_channel: int = 0,
    model_type: str = "nuclei",
    pretrained_model: Union[str, Path, None] = None,
    diameter: Optional[int] = None,
    threshold: float = 0,
    flow_threshold: float = 0.4,
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
    holes = np.logical_xor(mh.close_holes(foreground_mask),foreground_mask)
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
            raise ValueError(
                f"Objects are not consecutively labeled, please relabel before secondary segmentation."
            )
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
                raise ValueError(
                    f"Number of objects ({n_objects}) exceeds 32-bit datatype."
                )
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


def _get_channel_names(image: AICSImage, input: Optional[Union[int, str, List[Union[int, str]]]] = None) -> List[str]:
    """
    Get the channel names based on input.

    Parameters
    ----------
    image
        An instance of AICSImage with channel_names attribute.
    input
        The input specifying channels as integer(s) or string(s).
        Defaults to None.

    Returns
    -------
    A list of strings representing the requested channel names, without duplicates,
    or the full list of channel names if input is None.

    Raises
    ------
        ValueError: If input is not of appropriate type or contains invalid channel names.
    """

    if input is None:
        return image.channel_names

    if not isinstance(input, (int, str, list)):
        raise ValueError("Input must be an integer, a string, a list of integers/strings, or None.")

    if isinstance(input, int):
        if input not in range(len(image.channel_names)):
            raise ValueError(
                f"Integer input for channel name must be in the range 0 to {len(image.channel_names)}."
            )
        return [image.channel_names[input]]

    elif isinstance(input, str):
        if input not in image.channel_names:
            raise ValueError("String input for channel name must be a valid channel name.")
        return [input]

    elif isinstance(input, list):
        channel_names = []
        for item in input:
            if isinstance(item, int):
                if item not in range(len(image.channel_names)):
                    raise ValueError(
                        f"Integer input for channel name must be in the range 0 to {len(image.channel_names)}."
                    )
                channel_names.append(image.channel_names[item])
            elif isinstance(item, str):
                if item not in image.channel_names:
                    raise ValueError("String input must be a valid channel name.")
                channel_names.append(item)
            else:
                raise ValueError("List elements must be integers or strings.")
        return list(set(channel_names))  # Remove duplicates


def cropped_array_containing_object(array: np.ndarray, bboxes: list, label: int, pad: int=1) -> np.ndarray:
    """
    Extract a region from the input array corresponding to a single object.

    Parameters
    ----------
    array
        The input array from which the object intensity will be extracted.
    bboxes
        List containing [start_row, end_row, start_column, end_column] of the bounding box.
    label
        The label of the object for which the intensity needs to be extracted.

    Returns
    -------
    numpy.ndarray
        Intensity values of the specified object extracted from the intensity image.

    Raises
    ------
    ValueError
        If the array is None.

    Notes
    -----
    The function uses the provided bounding box coordinates to extract the region of
    interest (ROI) corresponding to the specified object label. It then optionally pads
    the extracted ROI and returns it as a separate image.

    Examples
    --------
    >>> array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> bboxes = {'object1': [0, 2, 0, 2]}
    >>> label = 'object1'
    >>> object_intensity = get_object_array(array, bboxes, label)
    >>> print(object_intensity)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    """
    if array is None:
        raise ValueError("No intensity image available.")
    bbox = bboxes[label]
    return extract_bbox(array, bbox=bbox, pad=pad)


def extract_bbox(array: np.ndarray, bbox: list, pad: int = 0) -> np.ndarray:
    """
    Extracts a bounding box region from the given image.

    Parameters
    ----------
    array : numpy.ndarray
        The input image from which the bounding box region will be extracted.
    bbox : list
        List containing [start_row, end_row, start_column, end_column] of the bounding box.
    pad : int, optional
        The number of pixels to pad around the extracted region, by default 0.

    Returns
    -------
    numpy.ndarray
        The extracted bounding box region from the input image.

    Notes
    -----
    This function extracts the region defined by the provided bounding box coordinates
    from the input image. Optionally, padding can be applied around the extracted region.

    Examples
    --------
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> bbox = [0, 1, 0, 2]
    >>> padded_bbox = extract_bbox(image, bbox, pad=1)
    >>> print(padded_bbox)
    [[0 0 0 0 0]
     [0 1 2 3 0]
     [0 4 5 6 0]
     [0 0 0 0 0]]
    """
    cropped_array = array[bbox[0] : bbox[1], bbox[2] : bbox[3]]
    if pad:
        cropped_array = np.lib.pad(cropped_array, (pad, pad), "constant", constant_values=(0))
    return cropped_array


def _object_mips_3D(
    intensity_array: np.ndarray,
    label_array: np.ndarray,
) -> List:
    """
    Generate Maximum Intensity Projections (MIPs) for labeled 3D regions.

    This function computes the MIP for each labeled region in a 3D intensity array.
    It identifies individual regions based on the provided label array, crops the
    corresponding regions from the intensity array, masks out regions not belonging 
    to the current label, and then calculates the MIP along the z-axis for each region.

    Parameters
    ----------
    intensity_array : np.ndarray
        A 3D numpy array representing the intensity values of the volume.
    label_array : np.ndarray
        A 3D numpy array with the same shape as `intensity_array`, where each unique
        integer label identifies a different region in the volume.

    Returns
    -------
    List
        A list of 2D numpy arrays, each representing the MIP of a labeled region along 
        the z-axis.
    """

    regions = skimage.measure.regionprops(label_array)
    mips = [None] * len(regions)

    for index, region in enumerate(regions):
        bbox = region.bbox
        object_crop = intensity_array[
            bbox[0]:bbox[3],
            bbox[1]:bbox[4],
            bbox[2]:bbox[5]
        ]
        mask_crop = label_array[
            bbox[0]:bbox[3],
            bbox[1]:bbox[4],
            bbox[2]:bbox[5]
        ]
        object_crop_masked = np.copy(object_crop)
        object_crop_masked[mask_crop!=index+1] = 0
        mips[index] = object_crop_masked.max(axis=0)
    return mips


def _object_middle_planes_3D(
    intensity_array: np.ndarray,
    label_array: np.ndarray
) -> List:
    """
    Extract middle planes of labeled 3D regions.

    This function computes the middle plane for each labeled region in a 3D intensity array.
    It identifies individual regions based on the provided label array, determines the middle
    plane along the z-axis for each region, crops the corresponding region from the intensity
    array, masks out regions not belonging to the current label, and then stores the masked 
    middle plane.

    Parameters
    ----------
    intensity_array : np.ndarray
        A 3D numpy array representing the intensity values of the volume.
    label_array : np.ndarray
        A 3D numpy array with the same shape as `intensity_array`, where each unique
        integer label identifies a different region in the volume.

    Returns
    -------
    List
        A list of 2D numpy arrays, each representing the masked middle plane of a labeled
        region along the z-axis.
    """
    regions = skimage.measure.regionprops(label_array)
    meds = [None] * len(regions)
    for index, region in enumerate(regions):
        bbox = region.bbox
        object_crop = intensity_array[
            int(np.rint((bbox[0]+bbox[3])/2.)),
            bbox[1]:bbox[4],
            bbox[2]:bbox[5]
        ]
        mask_crop = label_array[
            int(np.rint((bbox[0]+bbox[3])/2.)),
            bbox[1]:bbox[4],
            bbox[2]:bbox[5]
        ]
        object_crop_masked = np.copy(object_crop)
        object_crop_masked[mask_crop!=index+1] = 0
        meds[index] = object_crop_masked
    return meds


def concatenated_projection_image_3D(
    intensity_image: AICSImage, 
    label_image: AICSImage, 
    label_channel: int = 0, 
    label_name: str = "Nuclei-3D-MIP",
    projection_type: str = "MIP"
) -> Tuple[AICSImage, AICSImage]:
    """
    Generate concatenated projections (MIPs or middle planes) for 3D intensity and label images.

    This function computes either the Maximum Intensity Projections (MIPs) or middle planes 
    for the labeled regions in a 3D intensity image, and concatenates them to form a single 
    2D image per channel. It also creates a concatenated projection image for the label image itself.

    This can be used to compute intensity feature values using the standard quantification pipeline
    for 2D images.

    Parameters
    ----------
    intensity_image : AICSImage
        AICSImage object containing the 3D intensity data.
    label_image : AICSImage
        AICSImage object containing the 3D label data.
    label_channel : int, optional
        The channel index in `label_image` to be used for generating the label projection, 
        by default 0.
    projection_type : str, optional
        Type of projection to generate. Either "MIP" for Maximum Intensity Projection 
        or "middle" for middle plane projection, by default "MIP".

    Returns
    -------
    Tuple[AICSImage, AICSImage]
        A tuple containing two AICSImage objects: the concatenated projection of the intensity 
        image and the concatenated projection of the label image.

    Raises
    ------
    ValueError
        If the dimensions of `label_image` and `intensity_image` do not match.
        If either `label_image` or `intensity_image` has Z-dimension size <= 1.
        If `projection_type` is not "mip" or "middle".
    """

    # Check if dimensions of intensity_image and label_image match
    if label_image.dims[["T", "Z", "Y", "X"]] != intensity_image.dims[["T", "Z", "Y", "X"]]:
        raise ValueError("The dimensions of label_image and intensity_image do not match.")
    
    # Check if the Z-dimension is greater than 1
    if label_image.dims.Z <= 1:
        raise ValueError("Both label_image and intensity_image must have a Z-dimension greater than 1.")

    # Determine the projection function and label name based on projection_type
    if projection_type == "MIP":
        projection_func = _object_mips_3D
        label_name = label_name
    elif projection_type == "middle":
        projection_func = _object_middle_planes_3D
        label_name = label_name
    else:
        raise ValueError("projection_type must be either 'mip' or 'middle'.")

    # Generate label projection arrays
    label_projection_arrays = projection_func(
        intensity_array=label_image.get_image_data('ZYX', C=label_channel),
        label_array=label_image.get_image_data('ZYX', C=label_channel)
    )

    y_extent = max(m.shape[0] for m in label_projection_arrays)
    
    # Concatenate label projection arrays
    concat_label_projection_arrays = np.concatenate(
        [np.pad(m, pad_width=(y_extent - m.shape[0], 0), constant_values=0) for m in label_projection_arrays],
        axis=1
    )

    projection_label_image = AICSImage(
        concat_label_projection_arrays[np.newaxis, np.newaxis, np.newaxis, :, :],
        channel_names=[label_name],
        physical_pixel_sizes=label_image.physical_pixel_sizes,
    )

    # Generate intensity projection arrays and concatenate them
    concat_intensity_projection_arrays = [
        np.concatenate(
            [
                np.pad(m, pad_width=(y_extent - m.shape[0], 0), constant_values=0) 
                for m in projection_func(
                    intensity_array=intensity_image.get_image_data('ZYX', C=channel),
                    label_array=label_image.get_image_data('ZYX', C=label_channel),
                )
            ],
            axis=1
        ) 
        for channel in range(intensity_image.dims.C)
    ]
    
    concat_intensity_projection_stack = np.stack(concat_intensity_projection_arrays, axis=0)
    
    projection_intensity_image = AICSImage(
        concat_intensity_projection_stack[np.newaxis, :, np.newaxis, :, :],
        channel_names=intensity_image.channel_names,
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    return projection_intensity_image, projection_label_image


def _calculate_texture_features_single_object(
    intensity_array: np.ndarray, channel_name: str, object_name: str, bboxes: list, label: int, scales: List[int]
) -> pd.DataFrame:
    """
    Calculate texture features for a single object from an intensity array.

    Parameters
    ----------
    intensity_array
        The input intensity array containing image data.
    channel_name
        The name of the channel being processed.
    object_name
        The name of the object for which texture features are calculated.
    bboxes
        List of bounding box coordinates for different objects.
    label
        The label of the object for which texture features are calculated.
    scales
        List of scales used for calculating Haralick texture features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing Haralick texture features for the specified object,
        channel, and different scales.

    Notes
    -----
    This function calculates Haralick texture features for a single object and a
    specified channel from the given intensity array. It rescales the intensity array,
    calculates Haralick features for each scale, and constructs a DataFrame containing
    the features for the given object, channel, and different scales.
    """

    # extract the intensity image for the current object
    cropped_intensity_array = cropped_array_containing_object(array=intensity_array, bboxes=bboxes, label=label)

    # compute texture features on rescaled (8-bit) image
    clip_value = np.percentile(cropped_intensity_array, 99.9)
    rescaled_cropped_intensity_array = mh.stretch(np.clip(cropped_intensity_array, 0, clip_value))

    haralick_df_list_multiple_scales = []
    for scale in scales:
        try:
            haralick_values = mh.features.haralick(
                rescaled_cropped_intensity_array, ignore_zeros=True, return_mean=True, distance=scale
            )

        except ValueError:
            haralick_values = np.full(len(HARALICK_BASE_NAMES), np.NaN, dtype=float)

        if not isinstance(haralick_values, np.ndarray):
            # NOTE: setting `ignore_zeros` to True creates problems for some
            # objects, when all values of the adjacency matrices are zeros
            haralick_values = np.full(len(HARALICK_BASE_NAMES), np.NaN, dtype=float)

        # adjust feature names using object and scale
        haralick_full_names = [
            "{object_name}_{channel_name}_Haralick-{name}-{scale}".format(
                object_name=object_name, channel_name=channel_name, name=name, scale=scale
            )
            for name in HARALICK_BASE_NAMES
        ]

        # convert results to dataframe row and add current object label
        df = pd.DataFrame(haralick_values.reshape(1, -1), columns=haralick_full_names)
        df["label"] = label
        haralick_df_list_multiple_scales.append(df.set_index("label"))

    # collect multiple scales for this object
    haralick_df = reduce(
        lambda left, right: pd.merge(left, right, on=["label"], how="outer"), haralick_df_list_multiple_scales
    )

    return haralick_df


def border_objects(label_array: np.ndarray) -> pd.DataFrame:
    all_object_labels = [x for x in np.unique(label_array) if x != 0]

    label_array_non_border = clear_border(label_array)
    non_border_object_labels = np.unique(label_array_non_border)

    border_objects = pd.DataFrame({"label": all_object_labels})
    border_objects["is_border"] = border_objects.label.map(lambda x: False if x in non_border_object_labels else True)

    return border_objects


def border_objects_XY_3D(label_image, label_channel=0):
    label_array = label_image.get_image_data('ZYX', C=label_channel)
    all_object_labels = [x for x in np.unique(label_array) if x != 0]

    all_border_object_labels = set()

    for z in range(label_image.dims.Z):
        label_array_plane = label_image.get_image_data('YX', Z=z, C=label_channel)
        all_object_labels_plane = [x for x in np.unique(label_array_plane) if x != 0]

        label_array_XY_non_border = clear_border(label_array_plane)
        border_object_labels = set(all_object_labels_plane) - set(np.unique(label_array_XY_non_border))
        all_border_object_labels.update(border_object_labels)

    border_objects = pd.DataFrame({'label': all_object_labels})
    border_objects['is_border_XY'] = border_objects['label'].isin(all_border_object_labels)

    return border_objects


def _quantify_single_timepoint(
    intensity_image: AICSImage,
    label_image: AICSImage,
    timepoint: int = 0,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    intensity_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> pd.DataFrame:
    """Quantify all channels in an image relative to a matching label image.
    Single time-point only.

    Parameters
    ----------
    intensity_image
        intensity image (possibly 5D: "TCZYX")
    label_image
        label image (possibly 5D: "TCZYX")
    timepoint
        which timepoint should be quantified
    intensity_channels
        channels in ``intensity_image`` to be used for intensity calculations,
        can be provided as indices or names (see ``_get_channel_names()``)
    intensity_objects
        objects in ``intensity_image`` to be used for intensity calculations,
        can be provided as indices or names (see ``_get_channel_names()``)
    texture_channels
        channels in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``_get_channel_names()``)
    texture_objects
        objects in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``_get_channel_names()``)
    texture_scales
        length scales at which to calculate textures

    Returns
    -------
    pandas.DataFrame
        quantified data (n_rows = # objects, n_cols = # features)
    """

    features_list = []

    def intensity_sd(regionmask, intensity_image):
        return np.std(intensity_image[regionmask])

    def intensity_median(regionmask, intensity_image):
        return np.median(intensity_image[regionmask])

    intensity_channels_list = _get_channel_names(intensity_image, intensity_channels)
    intensity_objects_list = _get_channel_names(label_image, intensity_objects)
    if calculate_textures:
        texture_channels_list = _get_channel_names(intensity_image, texture_channels)
        texture_objects_list = _get_channel_names(label_image, texture_objects)

    # iterate over all object types in the segmentation
    for obj_index, obj in enumerate(label_image.channel_names):
        label_array = label_image.get_image_data("YX", C=obj_index, T=timepoint, Z=0)

        # Morphology features
        # -----------------------
        features = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_array,
                properties=[
                    "label",
                    "centroid",
                    "area",
                    "area_convex",
                    "axis_major_length",
                    "axis_minor_length",
                    "eccentricity",
                    "extent",
                    "feret_diameter_max",
                    "solidity",
                    "perimeter",
                    "perimeter_crofton",
                    "euler_number",
                ],
                separator="_",
            )
        ).rename(columns=lambda x: obj + "_" + x if x != "label" else x)

        features = features.merge(
            border_objects(label_array).rename(columns=lambda x: obj + "_" + x if x != "label" else x), on="label"
        )

        # Intensity features
        # ----------------------
        # iterate over selected channels
        for channel in intensity_channels_list:
            channel_index = intensity_image.channel_names.index(channel)
            intensity_array = intensity_image.get_image_data("YX", C=channel_index, T=timepoint, Z=0)

            if obj in intensity_objects_list:
                intensity_features = pd.DataFrame(
                    skimage.measure.regionprops_table(
                        label_array,
                        intensity_array,
                        properties=["label", "intensity_mean"],
                        extra_properties=(intensity_sd, intensity_median),
                        separator="_",
                    )
                ).rename(columns=lambda x: obj + "_" + x + "_" + channel if x != "label" else x)

                features = features.merge(intensity_features, on="label")

        # Texture features
        # ----------------------
        # iterate over selected channels
        if calculate_textures:
            for channel in texture_channels_list:
                channel_index = intensity_image.channel_names.index(channel)
                intensity_array = intensity_image.get_image_data("YX", C=channel_index, T=timepoint, Z=0)

                if obj in texture_objects_list:
                    bboxes = mh.labeled.bbox(label_array)
                    texture_features_list = [
                        _calculate_texture_features_single_object(
                            intensity_array=intensity_array,
                            channel_name=channel,
                            object_name=obj,
                            bboxes=bboxes,
                            label=label,
                            scales=texture_scales,
                        )
                        for label in np.unique(label_array[label_array != 0])
                    ]

                    # collect data for all objects and merge with morphology/intensity features
                    texture_features = pd.concat(texture_features_list, ignore_index=False)
                    features = features.merge(texture_features, on="label")

        features_list.append(features)

    # combine results for all objects (assumes matching labels)
    # TODO: generalise this for aggregate quantification, etc.
    features = reduce(
        lambda left, right: pd.merge(left, right, on=["label"], how="outer"),
        features_list,
    )

    # add timepoint information (note + 1 to match metadata)
    features[["TimepointID"]] = timepoint + 1
    return features


def _quantify_single_timepoint_3D(
    intensity_image: AICSImage,
    label_image: AICSImage,
    timepoint: int = 0,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    intensity_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> pd.DataFrame:
    """Quantify all channels in an image relative to a matching label image.
    Single time-point only. For 3D images, this is a wrapper that sits above 
    the standard 2D feature extraction function.

    FIXME: this is unfinished

    """

    if (intensity_image.physical_pixel_sizes is None or
        intensity_image.physical_pixel_sizes.Z is None or
        intensity_image.physical_pixel_sizes.Y is None or
        intensity_image.physical_pixel_sizes.X is None):
        raise ValueError(
            "intensity_image has undetermined physical_pixel_sizes. Cannot calculate 3D morphology features."
        )

    features_list = []

    def intensity_sd(regionmask, intensity_image):
        return np.std(intensity_image[regionmask])

    def intensity_median(regionmask, intensity_image):
        return np.median(intensity_image[regionmask])

    intensity_channels_list = _get_channel_names(intensity_image, intensity_channels)
    intensity_objects_list = _get_channel_names(label_image, intensity_objects)

    # iterate over all object types in the segmentation
    for obj_index, obj in enumerate(label_image.channel_names):
        label_array = label_image.get_image_data("ZYX", C=obj_index, T=timepoint)

        # Morphology features
        # -----------------------
        morphology_features_3D = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_array,
                spacing=(
                    intensity_image.physical_pixel_sizes.Z/1.0e6,
                    intensity_image.physical_pixel_sizes.Y/1.0e6,
                    intensity_image.physical_pixel_sizes.X/1.0e6
                ),
                properties=[
                    "label",
                    "centroid",
                    "area",
                    "area_convex",
                    "axis_major_length",
                    "axis_minor_length",
                    "extent",
                    "feret_diameter_max",
                    "solidity",
                ],
                separator="_",
            )
        ).rename(columns=lambda x: obj + "_3D_" + x if x != "label" else x)

        # Intensity features
        # ----------------------
        # iterate over selected channels
        for channel in intensity_channels_list:
            channel_index = intensity_image.channel_names.index(channel)
            intensity_array = intensity_image.get_image_data("ZYX", C=channel_index, T=timepoint)

            if obj in intensity_objects_list:
                intensity_features_3D = pd.DataFrame(
                    skimage.measure.regionprops_table(
                        label_array,
                        intensity_array,
                        properties=["label", "intensity_mean"],
                        extra_properties=(intensity_sd, intensity_median),
                        separator="_",
                    )
                ).rename(columns=lambda x: obj + "_3D_" + x + "_" + channel if x != "label" else x)

                features_3D = morphology_features_3D.merge(intensity_features_3D, on="label")

        # Object MIP features
        # ----------------------------
        # Use maximum-intensity projection to isolate a 2D image from each 3D object.
        # Areas outside the objects are masked. 
        intensity_image_object_mip, label_image_object_mip = concatenated_projection_image_3D(
            intensity_image,
            label_image,
            label_name=obj + "-3D-MIP",
            projection_type="MIP")

        object_mip_features = _quantify_single_timepoint(
            intensity_image=intensity_image_object_mip,
            label_image=label_image_object_mip,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            intensity_objects=intensity_objects,
            calculate_textures=calculate_textures,
            texture_channels=texture_channels,
            texture_objects=texture_objects,
            texture_scales=texture_scales
        )
        # eliminate centroid and border features, which are meaningless in a
        # concatenated image, and TimepointID, which we add later to avoid 
        # duplication
        object_mip_features.drop(
            list(object_mip_features.filter(regex='centroid|border|TimepointID')),
            axis=1, inplace=True)

        # Object middle Z-plane features
        # -----------------------
        intensity_image_object_middle, label_image_object_middle = concatenated_projection_image_3D(
            intensity_image,
            label_image,
            label_name=obj + "-3D-Middle",
            projection_type="middle")

        object_middle_features = _quantify_single_timepoint(
            intensity_image=intensity_image_object_middle,
            label_image=label_image_object_middle,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            intensity_objects=intensity_objects,
            calculate_textures=calculate_textures,
            texture_channels=texture_channels,
            texture_objects=texture_objects,
            texture_scales=texture_scales
        )
        # eliminate centroid and border features, which are meaningless in a
        # concatenated image, and TimepointID, which we add later to avoid 
        # duplication
        object_middle_features.drop(
            list(object_mip_features.filter(regex='centroid|border|TimepointID')),
            axis=1,
            inplace=True
        )

        # Border features
        # ---------------
        # Is an object touching the 3D border?
        border_3D = border_objects(
            label_image.get_image_data('ZYX', C=channel_index)
        ).rename(columns=lambda x: "Nuclei_3D_" + x if x != "label" else x)

        # Is an object touching the XY border?
        border_XY_3D = border_objects_XY_3D(
            label_image,
            label_channel=channel_index
            ).rename(columns=lambda x: "Nuclei_" + x if x != "label" else x)

        # Merge all features
        # ----------------------
        features = reduce(
            lambda left, right: pd.merge(left, right, on='label', how='outer'),
            [features_3D,
             object_mip_features,
             object_middle_features,
             border_3D,
             border_XY_3D]
        )

        features_list.append(features)

    # combine results for all objects (assumes matching labels)
    # TODO: generalise this for aggregate quantification, etc.
    features = reduce(
        lambda left, right: pd.merge(left, right, on=["label"], how="outer"),
        features_list,
    )

    # add timepoint information (note + 1 to match metadata)
    features[["TimepointID"]] = timepoint + 1

    return features


def quantify(
    intensity_image: AICSImage,
    label_image: AICSImage,
    timepoint: Union[int, None] = None,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    intensity_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
):
    """Quantify all channels in an image relative to a matching segmentation
    label image.

    Parameters
    ----------
    intensity_image
        intensity image (possibly 5D: "TCZYX")
    label_image
        label image (possibly 5D: "TCZYX")
    timepoint
        which timepoint should be segmented (optional,
        default None will segment all time-points)

    Returns
    -------
    pandas.DataFrame
        quantified data (n_rows = # objects x # timepoints, n_cols = # features)
    """

    if timepoint is None:
        if intensity_image.dims.Z > 1:
            # 3D quantification
            features = pd.concat(
                [
                    _quantify_single_timepoint_3D(
                        intensity_image=intensity_image,
                        label_image=label_image,
                        timepoint=t,
                        intensity_channels=intensity_channels,
                        intensity_objects=intensity_objects,
                        calculate_textures=calculate_textures,
                        texture_channels=texture_channels,
                        texture_objects=texture_objects,
                        texture_scales=texture_scales,
                    )
                    for t in range(intensity_image.dims[["T"]][0])
                ]
            )
        else:
            # 2D quantification
            features = pd.concat(
                [
                    _quantify_single_timepoint(
                        intensity_image=intensity_image,
                        label_image=label_image,
                        timepoint=t,
                        intensity_channels=intensity_channels,
                        intensity_objects=intensity_objects,
                        calculate_textures=calculate_textures,
                        texture_channels=texture_channels,
                        texture_objects=texture_objects,
                        texture_scales=texture_scales,
                    )
                    for t in range(intensity_image.dims[["T"]][0])
                ]
            )
    else:
        if intensity_image.dims.Z > 1:
            # 3D quantification
            features = _quantify_single_timepoint_3D(intensity_image, label_image, timepoint)
        else:
            # 2D quantification
            features = _quantify_single_timepoint(intensity_image, label_image, timepoint)
    return features


def segment_and_quantify(
    image_file: Union[str, Path],
    nuclei_channel: int = 0,
    metadata_file: Union[str, Path, None] = None,
    timepoint: Union[int, None] = None,
):
    """
    Segment objects and quantify intensities of all channels relative to objects.

    Parameters
    ----------
    image_file
        Path to the input image file.
    nuclei_channel
        Index of the nuclei channel in the input image, by default 0.
    metadata_file
        Path to the metadata file or None to use the default location, by default None.
    timepoint
        Timepoint to process or None for all timepoints, by default None.

    Returns
    -------
    Tuple[AICSImage, pandas.DataFrame]
        A tuple containing the nuclei label image and a DataFrame with quantified features.

    Notes
    -----
    This function segments objects using the specified nuclei channel and then quantifies
    intensities of all channels relative to the segmented objects. The results are saved as
    label images and features in corresponding directories. The segmented nuclei label image
    and the quantified features DataFrame are returned as a tuple.

    Examples
    --------
    >>> image_file = "path/to/image.tif"
    >>> nuclei_channel = 0
    >>> metadata_file = "path/to/metadata.pkl"
    >>> timepoint = 1
    >>> nuclei_label_image, features_df = segment_and_quantify(image_file, nuclei_channel, metadata_file, timepoint)
    >>> print(features_df.head())
       ObjectID  Channel0  Channel1  ...  MetadataField1  MetadataField2  MetadataField3
    0         1     123.4     567.8  ...             1.0             2.0             3.0
    1         2     234.5     678.9  ...             1.0             2.0             3.0
    ...       ...       ...       ...  ...             ...             ...             ...
    """
    from blimp.preprocessing.operetta_parse_metadata import load_image_metadata

    # read intensity image and metadata
    intensity_image = AICSImage(image_file)

    if metadata_file is None:
        metadata_file = Path(image_file).parent / "image_metadata.pkl"
        logging.warning(f"Metadata file not provided, using default location: {str(metadata_file)}")
    else:
        metadata_file = Path(metadata_file)

    if not metadata_file.exists():
        logging.error(f"No metadata file provided: {str(metadata_file)} does not exist")
    metadata = load_image_metadata(metadata_file)

    # make label image directory
    label_image_dir = Path(image_file).parent / "LABEL_IMAGE"
    label_image_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(image_file).parent / "FEATURES"
    features_dir.mkdir(parents=True, exist_ok=True)

    # segment
    nuclei_label_image = segment_nuclei_cellpose(
        intensity_image=intensity_image,
        nuclei_channel=nuclei_channel,
        timepoint=timepoint,
    )
    nuclei_label_image.save(label_image_dir / Path("nuclei_" + Path(image_file).name))

    # quantify intensities relative to masks
    features = quantify(
        intensity_image=intensity_image,
        label_image=nuclei_label_image,
        timepoint=timepoint,
    )

    # combine with metadata and save as csv
    features = features.merge(
        right=metadata[metadata["URL"] == Path(image_file).name],
        how="left",
        on="TimepointID",
    )
    features.to_csv(features_dir / Path(Path(image_file).stem + ".csv"), index=False)

    return (nuclei_label_image, features)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="segment_and_quantify")

    parser.add_argument("-i", "--image_file", help="full path to the image file", required=True)

    parser.add_argument("--nuclei_channel", default=0, help="channel nuber for nuclei", required=True)

    parser.add_argument("-m", "--metadata_file", default=None, help="full path to the metadata file")

    parser.add_argument(
        "-t",
        "--timepoint",
        default=None,
        help="analyse only one timepoint (enter number, 0-index)",
        required=True,
    )

    args = parser.parse_args()

    segment_and_quantify(
        image_file=args.image_file,
        nuclei_channel=args.nuclei_channel,
        metadata_file=args.metadata_file,
        timepoint=args.timepoint,
    )
