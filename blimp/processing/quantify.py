from typing import List, Union, Optional
from functools import reduce

from aicsimageio import AICSImage
from skimage.segmentation import clear_border
import numpy as np
import pandas as pd
import mahotas as mh
import skimage.measure

from blimp.utils import (
    get_channel_names,
    cropped_array_containing_object,
    concatenated_projection_image_3D,
)

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
    label_array = label_image.get_image_data("ZYX", C=label_channel)
    all_object_labels = [x for x in np.unique(label_array) if x != 0]

    all_border_object_labels = set()

    for z in range(label_image.dims.Z):
        label_array_plane = label_image.get_image_data("YX", Z=z, C=label_channel)
        all_object_labels_plane = [x for x in np.unique(label_array_plane) if x != 0]

        label_array_XY_non_border = clear_border(label_array_plane)
        border_object_labels = set(all_object_labels_plane) - set(np.unique(label_array_XY_non_border))
        all_border_object_labels.update(border_object_labels)

    border_objects = pd.DataFrame({"label": all_object_labels})
    border_objects["is_border_XY"] = border_objects["label"].isin(all_border_object_labels)

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
        can be provided as indices or names (see ``get_channel_names()``)
    intensity_objects
        objects in ``intensity_image`` to be used for intensity calculations,
        can be provided as indices or names (see ``get_channel_names()``)
    texture_channels
        channels in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``get_channel_names()``)
    texture_objects
        objects in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``get_channel_names()``)
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

    intensity_channels_list = get_channel_names(intensity_image, intensity_channels)
    intensity_objects_list = get_channel_names(label_image, intensity_objects)
    if calculate_textures:
        texture_channels_list = get_channel_names(intensity_image, texture_channels)
        texture_objects_list = get_channel_names(label_image, texture_objects)

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
                        properties=["label", "intensity_mean", "intensity_max", "intensity_min"],
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
    For use with 3D data and matching label images. Single time-point only.

    WARNING: 3D morphology features have not yet been thoroughly tested

    Textures are not calculate on 3D images, but rather on object-based
    maximum-intensity projections, and on the 2D image extracted from the
    "middle" (central-Z) plane of each object.

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
        can be provided as indices or names (see ``get_channel_names()``)
    intensity_objects
        objects in ``intensity_image`` to be used for intensity calculations,
        can be provided as indices or names (see ``get_channel_names()``)
    texture_channels
        channels in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``get_channel_names()``)
    texture_objects
        objects in ``intensity_image`` to be used for texture calculations,
        can be provided as indices or names (see ``get_channel_names()``)
    texture_scales
        length scales at which to calculate textures

    Returns
    -------
    pandas.DataFrame
        quantified data (n_rows = # objects, n_cols = # features)
    """

    if (
        intensity_image.physical_pixel_sizes is None
        or intensity_image.physical_pixel_sizes.Z is None
        or intensity_image.physical_pixel_sizes.Y is None
        or intensity_image.physical_pixel_sizes.X is None
    ):
        raise ValueError(
            "intensity_image has undetermined physical_pixel_sizes. Cannot calculate 3D morphology features."
        )

    features_list = []

    def intensity_sd(regionmask, intensity_image):
        return np.std(intensity_image[regionmask])

    def intensity_median(regionmask, intensity_image):
        return np.median(intensity_image[regionmask])

    intensity_channels_list = get_channel_names(intensity_image, intensity_channels)
    intensity_objects_list = get_channel_names(label_image, intensity_objects)

    if calculate_textures:
        texture_objects_list = get_channel_names(label_image, texture_objects)

    # iterate over all object types in the segmentation
    for obj_index, obj in enumerate(label_image.channel_names):
        label_array = label_image.get_image_data("ZYX", C=obj_index, T=timepoint)

        if calculate_textures:
            calculate_textures_for_this_object = obj in texture_objects_list
        else:
            calculate_textures_for_this_object = False

        # Morphology features
        # -----------------------
        morphology_features_3D = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_array,
                spacing=(
                    intensity_image.physical_pixel_sizes.Z / 1.0e6,
                    intensity_image.physical_pixel_sizes.Y / 1.0e6,
                    intensity_image.physical_pixel_sizes.X / 1.0e6,
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
                        properties=["label", "intensity_mean", "intensity_max", "intensity_min"],
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
            intensity_image, label_image, label_name=obj + "-3D-MIP", projection_type="MIP"
        )

        object_mip_features = _quantify_single_timepoint(
            intensity_image=intensity_image_object_mip,
            label_image=label_image_object_mip,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            intensity_objects=obj + "-3D-MIP",
            calculate_textures=calculate_textures_for_this_object,
            texture_channels=texture_channels,
            texture_objects=obj + "-3D-MIP",
            texture_scales=texture_scales,
        )
        # eliminate centroid and border features, which are meaningless in a
        # concatenated image, and TimepointID, which we add later to avoid
        # duplication
        object_mip_features.drop(
            list(object_mip_features.filter(regex="centroid|border|TimepointID")), axis=1, inplace=True
        )

        # Object middle Z-plane features
        # -----------------------
        intensity_image_object_middle, label_image_object_middle = concatenated_projection_image_3D(
            intensity_image, label_image, label_name=obj + "-3D-Middle", projection_type="middle"
        )

        object_middle_features = _quantify_single_timepoint(
            intensity_image=intensity_image_object_middle,
            label_image=label_image_object_middle,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            intensity_objects=obj + "-3D-Middle",
            calculate_textures=calculate_textures_for_this_object,
            texture_channels=texture_channels,
            texture_objects=obj + "-3D-Middle",
            texture_scales=texture_scales,
        )
        # eliminate centroid and border features, which are meaningless in a
        # concatenated image, and TimepointID, which we add later to avoid
        # duplication
        object_middle_features.drop(
            list(object_middle_features.filter(regex="centroid|border|TimepointID")), axis=1, inplace=True
        )

        # Border features
        # ---------------
        # Is an object touching the 3D border?
        border_3D = border_objects(label_image.get_image_data("ZYX", C=obj_index)).rename(
            columns=lambda x: obj + "_3D_" + x if x != "label" else x
        )

        # Is an object touching the XY border?
        border_XY_3D = border_objects_XY_3D(label_image, label_channel=obj_index).rename(
            columns=lambda x: obj + "_" + x if x != "label" else x
        )

        # Merge all features
        # ----------------------
        features = reduce(
            lambda left, right: pd.merge(left, right, on="label", how="outer"),
            [features_3D, object_mip_features, object_middle_features, border_3D, border_XY_3D],
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
