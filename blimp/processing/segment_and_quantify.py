from typing import List, Union, Optional
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
    pretrained_model: Union[str,Path,None] = None,
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
        cellpose_model = models.CellposeModel(gpu=False,pretrained_model=str(pretrained_model))
        masks, flows, styles = cellpose_model.eval(
            nuclei_images,
            channels = [0,0]
        )

    segmentation = AICSImage(
        np.stack(masks)[:, np.newaxis, np.newaxis, :],
        channel_names=["Nuclei"],
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    return segmentation


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
        if input not in image.channel_names:
            raise ValueError("Integer input must be a valid channel name.")
        return [image.channel_names[input]]

    elif isinstance(input, str):
        if input not in image.channel_names:
            raise ValueError("String input must be a valid channel name.")
        return [input]

    elif isinstance(input, list):
        channel_names = []
        for item in input:
            if isinstance(item, int):
                if item not in image.channel_names:
                    raise ValueError("Integer input must be a valid channel name.")
                channel_names.append(image.channel_names[item])
            elif isinstance(item, str):
                if item not in image.channel_names:
                    raise ValueError("String input must be a valid channel name.")
                channel_names.append(item)
            else:
                raise ValueError("List elements must be integers or strings.")
        return list(set(channel_names))  # Remove duplicates


def cropped_array_containing_object(array: np.ndarray, bboxes: list, label: int) -> np.ndarray:
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
    return extract_bbox(array, bbox=bbox, pad=1)


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
