import numpy as np
import pandas as pd
import skimage.measure
import logging
from functools import reduce
from aicsimageio import AICSImage
from cellpose import models
from typing import Union
from pathlib import Path


def segment_nuclei_cellpose(
    intensity_image: AICSImage,
    nuclei_channel: int = 0,
    threshold: float = 0,
    timepoint: Union[int, None] = None,
) -> AICSImage:
    """
    Segment nuclei

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

    if timepoint is None:
        nuclei_images = [
            intensity_image.get_image_data("ZYX", C=nuclei_channel, T=t)
            for t in range(intensity_image.dims[["T"]][0])
        ]
    else:
        nuclei_images = [
            intensity_image.get_image_data("ZYX", C=nuclei_channel, T=timepoint)
        ]

    cellpose_nuclei_model = models.Cellpose(gpu=False, model_type="nuclei")
    masks, flows, styles, diams = cellpose_nuclei_model.eval(
        nuclei_images,
        diameter=None,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=threshold,
        do_3D=False,
    )

    segmentation = AICSImage(
        np.stack(masks)[:, np.newaxis, np.newaxis, :],
        channel_names=["Nuclei"],
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    return segmentation


def _quantify_single_timepoint(
    intensity_image: AICSImage, label_image: AICSImage, timepoint: int = 0
) -> pd.DataFrame:
    """
    Quantify all channels in an image relative to a
    matching segmentation label image. Singel time-point
    only.

    Parameters
    ----------
    intensity_image
        intensity image (possibly 5D: "TCZYX")
    label_image
        label image (possibly 5D: "TCZYX")
    timepoint
        which timepoint should be quantified

    Returns
    -------
    pandas.DataFrame
        quantified data (n_rows = # objects, n_cols = # features)
    """

    features_list = []

    # iterate over all objects in the segmentation
    for obj_index, obj in enumerate(label_image.channel_names):

        # get morphology features
        # -----------------------
        features = pd.DataFrame(
            skimage.measure.regionprops_table(
                label_image.get_image_data("YX", C=obj_index, T=timepoint, Z=0),
                properties=["label", "centroid", "area"],
                separator="_",
            )
        ).rename(columns=lambda x: obj + "_" + x if x != "label" else x)

        # iterate over all channels in the image
        for channel_index, channel in enumerate(intensity_image.channel_names):

            # get intensity features
            # ----------------------
            intensity_features = pd.DataFrame(
                skimage.measure.regionprops_table(
                    label_image.get_image_data("YX", C=obj_index, T=timepoint, Z=0),
                    intensity_image.get_image_data(
                        "YX", C=channel_index, T=timepoint, Z=0
                    ),
                    properties=["label", "intensity_mean"],
                    separator="_",
                )
            ).rename(
                columns=lambda x: obj + "_" + x + "_" + channel if x != "label" else x
            )

            features = features.merge(intensity_features, on="label")

        features_list.append(features)

    # combine results for all objects (assumes matching labels)
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
):
    """
    Quantify all channels in an image relative to a
    matching segmentation label image.

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
                _quantify_single_timepoint(intensity_image, label_image, t)
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
) -> None:
    """
    Segment objects and quantify intensities of all channels
    relative to objects.
    """
    from blimp.preprocessing.operetta_parse_metadata import load_image_metadata

    # read intensity image and metadata
    intensity_image = AICSImage(image_file)

    if metadata_file is None:
        metadata_file = Path(image_file).parent / "image_metadata.pkl"
        logging.warning(
            "Metadata file not provided, using default location: {}".format(
                str(metadata_file)
            )
        )
    else:
        metadata_file = Path(metadata_file)

    if not metadata_file.exists():
        logging.error(
            "No metadata file provided: {} does not exist".format(str(metadata_file))
        )
    metadata = load_image_metadata(metadata_file)

    # make label image directory
    label_image_dir = Path(image_file).parent / "label_image"
    label_image_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(image_file).parent / "features"
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

    parser.add_argument(
        "-i", "--image_file", help="full path to the image file", required=True
    )

    parser.add_argument(
        "--nuclei_channel", default=0, help="channel nuber for nuclei", required=True
    )

    parser.add_argument(
        "-m", "--metadata_file", default=None, help="full path to the metadata file"
    )

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
