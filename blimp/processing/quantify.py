from typing import List, Union, Optional
from functools import reduce
import logging

from aicsimageio import AICSImage
from skimage.segmentation import clear_border
import numpy as np
import pandas as pd
import mahotas as mh
import skimage.measure

from blimp.utils import (
    get_channel_names,
    make_channel_names_unique,
    check_uniform_dimension_sizes,
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

logger = logging.getLogger(__name__)


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


def border_objects_XY_3D(label_image: AICSImage, label_channel: int = 0) -> pd.DataFrame:
    """Identify border objects in the XY plane across a 3D image stack and
    return a DataFrame indicating which objects touch the XY borders.

    Parameters
    ----------
    label_image : AICSImage
        3D labeled image where objects are represented by unique integer
        labels. It should contain multiple Z planes and at least one channel.
    label_channel : int, optional
        The channel index to extract from the image for analysis. Defaults to 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'label': The unique object labels found in the image.
        - 'is_border_XY': Boolean values indicating whether each object
          touches the border of any XY plane (True) or not (False).

    Notes
    -----
    Objects touching the XY border in any Z-plane are classified as
    border objects. Objects with label 0 (background) are excluded from the output.
    """

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


def _measure_parent_object_label(
    label_image: AICSImage, measure_object_index: int, parent_object_index: int, timepoint: int = 0
) -> pd.DataFrame:
    """
    Measure the parent object associated with each object in a labeled image.
    For each object in the `measure_object_index` channel, identify its corresponding
    parent object in the `parent_object_index` channel. This is based on the spatial
    overlap between the child and parent objects.

    Parameters
    ----------
    label_image : AICSImage
        The labeled image containing objects in separate channels, where each channel
        corresponds to a different object type (possibly 5D: "TCZYX").
    measure_object_index : int
        Index of the channel in `label_image` corresponding to the child objects
        that will be measured.
    parent_object_index : int
        Index of the channel in `label_image` corresponding to the parent objects
        to which the child objects are associated.
    timepoint : int, optional
        Timepoint at which to measure the objects, by default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'label': The label of each child object in the `measure_object_index` channel.
        - 'parent_label': The label of the corresponding parent object in the
          `parent_object_index` channel.
        - 'parent_label_name': The name of the parent object channel.

    Raises
    ------
    ValueError
        If any child object is associated with multiple parent objects (i.e., if the
        object is not fully contained within a single parent).

    Notes
    -----
    This function assumes that each child object is fully contained within a single
    parent object. If a child object overlaps with multiple parent objects, the function
    will raise an error.
    """

    if label_image.dims.Z == 1:
        logger.info("``label_image`` is 2D. Quantifying 2D features only.")
        label_array = label_image.get_image_data("YX", C=measure_object_index, T=timepoint, Z=0)
        parent_label_array = label_image.get_image_data("YX", C=parent_object_index, T=timepoint, Z=0)
    elif label_image.dims.Z > 1:
        logger.info(f"``label_image`` is 3D ({label_image.dims.Z} Z-planes). Measuring parent in 3D.")
        label_array = label_image.get_image_data("ZYX", C=measure_object_index, T=timepoint)
        parent_label_array = label_image.get_image_data("ZYX", C=parent_object_index, T=timepoint)

    # mask the parent object array using the label array
    parent_object_array_masked = np.where(label_array > 0, parent_label_array, 0)

    parent_id = pd.DataFrame(
        skimage.measure.regionprops_table(
            label_image=label_array,
            intensity_image=parent_object_array_masked,
            properties=[
                "label",
                "intensity_min",
                "intensity_max",
            ],
            separator="_",
        )
    )

    # check that each object has a unique parent id
    # (i.e. the child object is fully contained within the parent object)
    columns_match = (parent_id["intensity_min"] == parent_id["intensity_max"]).all()

    if not columns_match:
        raise ValueError(
            f"Measure objects ({label_image.channel_names[measure_object_index]}), not fully contained within parent objects ({label_image.channel_names[parent_object_index]})")

    parent_id["parent_label"] = np.floor(parent_id["intensity_max"]).astype(label_image.dtype)
    parent_id["parent_label_name"] = label_image.channel_names[parent_object_index]
    parent_id = parent_id.drop(["intensity_min", "intensity_max"], axis=1)

    return parent_id


def quantify_single_timepoint(
    intensity_image: AICSImage,
    label_image: AICSImage,
    measure_object: Union[int, str],
    parent_object: Optional[Union[int, str]] = None,
    timepoint: Optional[int] = None,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> pd.DataFrame:
    """Quantify all channels in an image relative to a matching label image.
    Single time-point only. Single object only.

    Parameters
    ----------
    intensity_image : AICSImage
        Intensity image (possibly 5D: "TCZYX").
    label_image : AICSImage
        Label image where objects are represented by unique integer labels
        (possibly 5D: "TCZYX").
    measure_object : Union[int, str]
        The object type (label) to be measured in the label image, either as an
        index or a string representing the object's channel name.
    parent_object : Optional[Union[int, str]], optional
        If provided, an optional parent object to associate with the measure_object
        for hierarchical measurements. Can be an index or a channel name, by default None.
    timepoint : int, optional
        The timepoint to quantify, by default 0.
    intensity_channels : Optional[Union[int, str, List[Union[int, str]]]], optional
        Channels in `intensity_image` to use for intensity calculations. Can be
        provided as indices or names. If None, no intensity features are calculated,
        by default None.
    calculate_textures : Optional[bool], optional
        Whether to calculate texture features, by default False.
    texture_channels : Optional[Union[int, str, List[Union[int, str]]]], optional
        Channels in `intensity_image` to use for texture calculations. Can be
        provided as indices or names. If None, no texture features are calculated,
        by default None.
    texture_scales : list, optional
        Length scales at which to calculate textures, by default [1, 3].

    Returns
    -------
    pd.DataFrame
        A DataFrame containing quantified data, with one row per object and one column per feature.

    Notes
    -----
    - Accepts 2D or 3D data as input.
    - For 3D data, textures are not calculate on 3D images, but rather on object-based
    maximum-intensity projections, and on the 2D image extracted from the
    "middle" (central-Z) plane of each object.
    """

    if intensity_image.dims.Z == 1:
        logger.info("``intensity_image`` is 2D. Quantifying 2D features only.")
        _quantify_single_timepoint_func = _quantify_single_timepoint_2D
    elif intensity_image.dims.Z > 1:
        logger.info(f"``intensity_image`` is 3D ({intensity_image.dims.Z} Z-planes). Quantifying 3D features.")
        _quantify_single_timepoint_func = _quantify_single_timepoint_3D

    if timepoint is None:
        features = pd.concat(
            [
                _quantify_single_timepoint_func(
                    intensity_image=intensity_image,
                    label_image=label_image,
                    measure_object=measure_object,
                    parent_object=parent_object,
                    timepoint=t,
                    intensity_channels=intensity_channels,
                    calculate_textures=calculate_textures,
                    texture_channels=texture_channels,
                    texture_scales=texture_scales,
                )
                for t in range(intensity_image.dims[["T"]][0])
            ]
        )
    else:
        features = _quantify_single_timepoint_func(
            intensity_image=intensity_image,
            label_image=label_image,
            measure_object=measure_object,
            parent_object=parent_object,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            calculate_textures=calculate_textures,
            texture_channels=texture_channels,
            texture_scales=texture_scales,
        )

    return features


def _quantify_single_timepoint_2D(
    intensity_image: AICSImage,
    label_image: AICSImage,
    measure_object: Union[int, str],
    parent_object: Optional[Union[int, str]] = None,
    timepoint: int = 0,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> pd.DataFrame:
    def intensity_sd(regionmask, intensity_image):
        return np.std(intensity_image[regionmask])

    def intensity_median(regionmask, intensity_image):
        return np.median(intensity_image[regionmask])

    # channels can be passed as names or indices, convert to names.
    intensity_channels_list = get_channel_names(intensity_image, intensity_channels)

    measure_object = get_channel_names(label_image, measure_object)[0]
    measure_object_index = label_image.channel_names.index(measure_object)
    if parent_object is not None:
        parent_object = get_channel_names(label_image, parent_object)[0]
        parent_object_index = label_image.channel_names.index(parent_object)

    if calculate_textures:
        texture_channels_list = get_channel_names(intensity_image, texture_channels)

    label_array = label_image.get_image_data("YX", C=measure_object_index, T=timepoint, Z=0)

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
    ).rename(columns=lambda x: measure_object + "_" + x if x != "label" else x)
    features = features.merge(
        border_objects(label_array).rename(columns=lambda x: measure_object + "_" + x if x != "label" else x),
        on="label",
    )
    # Intensity features
    # ----------------------
    # iterate over selected channels
    for channel in intensity_channels_list:
        channel_index = intensity_image.channel_names.index(channel)
        intensity_array = intensity_image.get_image_data("YX", C=channel_index, T=timepoint, Z=0)

        intensity_features = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_array,
                intensity_array,
                properties=["label", "intensity_mean", "intensity_max", "intensity_min"],
                extra_properties=(intensity_sd, intensity_median),
                separator="_",
            )
        ).rename(columns=lambda x: measure_object + "_" + x + "_" + channel if x != "label" else x)
        features = features.merge(intensity_features, on="label")

    # Texture features
    # ----------------------
    # iterate over selected channels
    if calculate_textures:
        for channel in texture_channels_list:
            channel_index = intensity_image.channel_names.index(channel)
            intensity_array = intensity_image.get_image_data("YX", C=channel_index, T=timepoint, Z=0)

            bboxes = mh.labeled.bbox(label_array)
            texture_features_list = [
                _calculate_texture_features_single_object(
                    intensity_array=intensity_array,
                    channel_name=channel,
                    object_name=measure_object,
                    bboxes=bboxes,
                    label=label,
                    scales=texture_scales,
                )
                for label in np.unique(label_array[label_array != 0])
            ]
            # collect data for all objects and merge with morphology/intensity features
            texture_features = pd.concat(texture_features_list, ignore_index=False)
            features = features.merge(texture_features, on="label")

    if parent_object is not None:
        parent_object_label = _measure_parent_object_label(label_image, measure_object_index, parent_object_index)
        features = features.merge(parent_object_label, on="label")

    # add timepoint information (note + 1 to match image metadata)
    features[["TimepointID"]] = timepoint + 1
    return features


def _quantify_single_timepoint_3D(
    intensity_image: AICSImage,
    label_image: AICSImage,
    measure_object: Union[int, str],
    parent_object: Optional[Union[int, str]] = None,
    timepoint: int = 0,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    calculate_textures: Optional[bool] = False,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> pd.DataFrame:
    if (
        intensity_image.physical_pixel_sizes is None
        or intensity_image.physical_pixel_sizes.Z is None
        or intensity_image.physical_pixel_sizes.Y is None
        or intensity_image.physical_pixel_sizes.X is None
    ):
        raise ValueError(
            "intensity_image has undetermined physical_pixel_sizes. Cannot calculate 3D morphology features."
        )

    def intensity_sd(regionmask, intensity_image):
        return np.std(intensity_image[regionmask])

    def intensity_median(regionmask, intensity_image):
        return np.median(intensity_image[regionmask])

    intensity_channels_list = get_channel_names(intensity_image, intensity_channels)

    measure_object = get_channel_names(label_image, measure_object)[0]
    measure_object_index = label_image.channel_names.index(measure_object)
    calculate_2D_derived = True
    if parent_object is not None:
        parent_object = get_channel_names(label_image, parent_object)[0]
        parent_object_index = label_image.channel_names.index(parent_object)
        logger.warning('Detecting parent objects in 3D leads to ambiguity for object relationships in derived 2D features.')
        logger.warning('Derived 2D features (3D-MIP and 3D-Middle) are omitted for clarity.')
        calculate_2D_derived = False

    if calculate_textures:
        get_channel_names(intensity_image, texture_channels)

    label_array = label_image.get_image_data("ZYX", C=measure_object_index, T=timepoint)

    # Morphology features
    # -----------------------
    morphology_features_3D = pd.DataFrame(
        skimage.measure.regionprops_table(
            label_array,
            spacing=(
                intensity_image.physical_pixel_sizes.Z,
                intensity_image.physical_pixel_sizes.Y,
                intensity_image.physical_pixel_sizes.X,
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
    ).rename(columns=lambda x: measure_object + "_3D_" + x if x != "label" else x)

    # Intensity features
    # ----------------------
    # iterate over selected channels
    for channel in intensity_channels_list:
        channel_index = intensity_image.channel_names.index(channel)
        intensity_array = intensity_image.get_image_data("ZYX", C=channel_index, T=timepoint)

        intensity_features_3D = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_array,
                intensity_array,
                properties=["label", "intensity_mean", "intensity_max", "intensity_min"],
                extra_properties=(intensity_sd, intensity_median),
                separator="_",
            )
        ).rename(columns=lambda x: measure_object + "_3D_" + x + "_" + channel if x != "label" else x)

        features_3D = morphology_features_3D.merge(intensity_features_3D, on="label")

    # Object MIP features
    # ----------------------------
    # Use maximum-intensity projection to isolate a 2D image from each 3D object.
    # Areas outside the objects are masked.
    if calculate_2D_derived:
        intensity_image_object_mip, label_image_object_mip = concatenated_projection_image_3D(
            intensity_image, label_image, label_name=measure_object + "-3D-MIP", projection_type="MIP"
        )

        object_mip_features = _quantify_single_timepoint_2D(
            intensity_image=intensity_image_object_mip,
            label_image=label_image_object_mip,
            measure_object=measure_object + "-3D-MIP",
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            calculate_textures=calculate_textures,
            texture_channels=texture_channels,
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
            intensity_image, label_image, label_name=measure_object + "-3D-Middle", projection_type="middle"
        )

        object_middle_features = _quantify_single_timepoint_2D(
            intensity_image=intensity_image_object_middle,
            label_image=label_image_object_middle,
            measure_object=measure_object + "-3D-Middle",
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            calculate_textures=calculate_textures,
            texture_channels=texture_channels,
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
    border_3D = border_objects(label_image.get_image_data("ZYX", C=measure_object_index)).rename(
        columns=lambda x: measure_object + "_3D_" + x if x != "label" else x
    )

    # Is an object touching the XY border?
    border_XY_3D = border_objects_XY_3D(label_image, label_channel=measure_object_index).rename(
        columns=lambda x: measure_object + "_" + x if x != "label" else x
    )

    # Merge all features
    # ----------------------
    if calculate_2D_derived:
        features = reduce(
            lambda left, right: pd.merge(left, right, on="label", how="outer"),
            [features_3D, object_mip_features, object_middle_features, border_3D, border_XY_3D],
        )
    else:
        features = reduce(
            lambda left, right: pd.merge(left, right, on="label", how="outer"),
            [features_3D, border_3D, border_XY_3D],
        )

    if parent_object is not None:
        parent_object_label = _measure_parent_object_label(label_image, measure_object_index, parent_object_index)
        features = features.merge(parent_object_label, on="label")

    # add timepoint information (note + 1 to match image metadata)
    features[["TimepointID"]] = timepoint + 1

    return features


def aggregate_and_merge_features(
    df_list: List[pd.DataFrame], parent_index: int, object_names: List[str]
) -> pd.DataFrame:
    """
    Aggregate feature data from multiple objects contained within 'parent' objects.
    Merge aggregated features with parent features and return a single dataframe.

    Parameters
    ----------
    df_list : List[pd.DataFrame]
        A list of feature tables

    parent_index : int
        The index of the parent feature table in `df_list`

    object_names : List[str]
        A list matching the length of `df_list` containing the object names (in the same order)

    Returns
    -------
    pd.DataFrame
        A feature table that results from merging the parent features with aggregated features from the other tables.

    Notes
    -----
    - The aggregation is performed on all numeric columns in the non-parent DataFrames, except for 'label',
      'TimepointID', and 'parent_label'.
    - Aggregation functions include 'mean', 'min', 'max', 'std', 'median' and 'count'
    - After aggregation, the result is merged with the parent DataFrame using the 'label' column from the parent
      and 'parent_label' from the aggregated data.
    """

    # check inputs
    if len(object_names) != len(df_list):
        raise ValueError(f"Expected {len(df_list)} object names but only {len(object_names)} were passed.")
    if parent_index >= len(object_names):
        raise ValueError(f"``parent_index`` = {parent_index} exceeds the length of ``df_list`` ({len(object_names)})")

    parent_df = df_list[parent_index]
    non_parent_dfs = [df for i, df in enumerate(df_list) if i != parent_index]
    non_parent_object_names = [obj for i, obj in enumerate(object_names) if i != parent_index]

    aggregations = ["mean", "min", "max", "std", "median"]

    # Initialize an empty list to store the aggregated dataframes
    aggregated_dfs = []
    for df_index, df in enumerate(non_parent_dfs):
        # Select only numeric columns for aggregation, excluding 'label', 'timepoint', and 'parent_label'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(
            ["label", "TimepointID", "parent_label"]
        )

        # Perform aggregation on numeric columns and flatten labels
        agg_df = df.groupby("parent_label").agg({col: aggregations for col in numeric_cols}).reset_index()
        agg_df.columns = ["_".join(col).strip() if col[0] != "parent_label" else col[0] for col in agg_df.columns]

        # add a count variable to quantify the number of objects aggregated
        count_var = non_parent_object_names[df_index] + "_count"
        object_count = df.groupby("parent_label").size().to_frame(count_var).reset_index()
        agg_df = agg_df.merge(object_count, how="left", left_on="parent_label", right_on="parent_label")

        aggregated_dfs.append(agg_df)

    # Merge all aggregated dataframes with the parent dataframe on parent_label -> label
    for agg_df in aggregated_dfs:
        out = parent_df.merge(agg_df, how="left", left_on="label", right_on="parent_label")

    # Replace NaN with 0 in columns that end with '_count'
    out.loc[:, out.columns.str.endswith("_count")] = out.loc[:, out.columns.str.endswith("count")].fillna(0).astype(int)

    return out


def quantify(
    intensity_image: AICSImage,
    label_image: AICSImage,
    measure_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    parent_object: Optional[Union[int, str]] = None,
    aggregate: Optional[bool] = False,
    timepoint: Optional[int] = None,
    intensity_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_channels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_objects: Optional[Union[int, str, List[Union[int, str]]]] = None,
    texture_scales: list = [1, 3],
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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

    # Check inputs
    check_uniform_dimension_sizes([label_image, intensity_image], omit="C", check_dtype=False)
    label_image = make_channel_names_unique(label_image)
    intensity_image = make_channel_names_unique(intensity_image)
    measure_objects = get_channel_names(image=label_image, input=measure_objects)
    texture_objects = get_channel_names(image=label_image, input=texture_objects)

    if aggregate and len(measure_objects) < 2:
        logger.warning(
            f"Cannot aggregate with only {len(measure_objects)} ``measure_objects``. Setting aggregate = False"
        )
        aggregate = False

    if parent_object is None:
        if aggregate:
            parent_object = label_image.channel_names[0]
            logger.warning(
                f"``parent_object`` not specified. Data will be aggregated relative to channel 0 ({parent_object})."
            )
        else:
            logger.debug(f"``parent_object`` not specified. Data will not be aggregated.")
    else:
        parent_object = get_channel_names(image=label_image, input=parent_object)[0]

    if aggregate and parent_object not in measure_objects:
        logger.info(f"Adding parent object to the list of measured objects.")
        measure_objects = measure_objects.append(parent_object)

    logger.info(f"``measure_objects`` =  {measure_objects}")
    logger.info(f"``texture_objects`` =  {texture_objects}")
    logger.info(f"``parent_object`` =  {parent_object}")

    # for each object, do the measurement
    features_list: List[pd.DataFrame] = []
    for obj_index, obj in enumerate(measure_objects):
        logger.debug(f"Quantifying {obj}")

        features = quantify_single_timepoint(
            intensity_image=intensity_image,
            label_image=label_image,
            measure_object=obj,
            parent_object=obj if parent_object is None else parent_object,
            timepoint=timepoint,
            intensity_channels=intensity_channels,
            calculate_textures=True if obj in texture_objects else False,
            texture_channels=texture_channels,
            texture_scales=texture_scales,
        )
        features_list.append(features)

    if aggregate:
        output = aggregate_and_merge_features(
            df_list=features_list,
            parent_index=label_image.channel_names.index(parent_object),
            object_names=measure_objects,
        )
    else:
        output = features_list

    return output
