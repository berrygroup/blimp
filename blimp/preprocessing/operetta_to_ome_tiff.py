"""Extract and parse metadata from Perkin-Elmer Operetta metadata files."""
from typing import List, Union, Pattern
from pathlib import Path
import os
import re
import logging

from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
import numpy as np
import pandas as pd

from blimp.log import configure_logging
from blimp.preprocessing.operetta_parse_metadata import (
    get_image_metadata,
    get_plate_metadata,
)

logger = logging.getLogger(__name__)

OPERETTA_REGEX = (
    r"r(?P<Row>\d+)c(?P<Col>\d+)f(?P<FieldID>\d+)p(?P<PlaneID>\d+)"
    + r"-ch(?P<ChannelID>\d+)sk(?P<TimepointID>\d+)(?P<End>fk\d+fl\d+)"
)
operetta_regex_filename_pattern = re.compile(OPERETTA_REGEX)


def _get_metadata_batch(image_metadata: pd.DataFrame, batch_id: int, n_batches: int) -> pd.DataFrame:
    """Get a predictable subset of dataframe rows from metadata.

    Parameters
    ----------
    image_metadata
        Metadata dataframe such as that generated by
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    batch_id
        0-indexed batch id
    n_batches
        how many batches the table should be split into

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = n_imaging_sites / n_batches)
    """

    image_metadata_subset = (
        image_metadata[["Row", "Col", "FieldID"]].drop_duplicates().sort_values(by=["Row", "Col", "FieldID"])
    )

    # convert batch_id and n_batches to rows of the metadata dataframe
    n_sites = image_metadata_subset.shape[0]
    n_sites_per_batch = int(np.ceil(float(n_sites) / float(n_batches)))
    row_min = int(batch_id) * n_sites_per_batch
    row_max = int(min(row_min + n_sites_per_batch, n_sites))
    logger.info(f"Splitting metadata into {n_batches} batches of size {n_sites_per_batch}")
    logger.info(f"Processing batch {batch_id}")

    if row_min >= row_max:
        logger.warning("batch_id exceeds metadata size, returning empty dataframe")

    return image_metadata_subset.iloc[row_min:row_max].merge(
        right=image_metadata, on=["Row", "Col", "FieldID"], how="left"
    )


def _read_images_single_site(
    in_path: Union[str, Path],
    image_metadata: pd.DataFrame,
    Row: int,
    Col: int,
    StandardFieldID: int,
    ChannelID: int,
    TimepointID: int,
) -> List[AICSImage]:
    """Load a set of image files for a single site.

    Reads a set of images corresponding to a single imaging site (field of
    view) from individual files on disk.

    Parameters
    ----------
    in_path
        Full path to the folder containing images
    image_metadata
        Metadata dataframe such as that generated by
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    Row
        Plate row
    Col
        Plate column
    StandardFieldID
        Field identifier (top-left to bottom-right,
        column-wise increasing first)
    ChannelID
        Channel identifier
    TimepointID
        Timepoint identifier

    Returns
    -------
    PhysicalPixelSizes
        AICSImage object for containing pixel dimensions
    """

    query = (
        "Row=="
        + str(Row)
        + " & Col=="
        + str(Col)
        + " & StandardFieldID=="
        + str(StandardFieldID)
        + " & ChannelID=="
        + str(ChannelID)
        + " & TimepointID=="
        + str(TimepointID)
    )

    # combine z and channel info into OME TIFFs
    # sort by increasing z position and and select a single channel/site
    filename_list = (
        image_metadata.sort_values(
            by=[
                "TimepointID",
                "ChannelID",
                "Row",
                "Col",
                "StandardFieldID",
                "AbsPositionZ",
            ]
        )
        .query(query)
        .URL.tolist()
    )
    filepath_list = [Path(in_path) / filename for filename in filename_list]
    images = [AICSImage(filepath) for filepath in filepath_list]
    return images


def _get_zyx_resolution(
    image_metadata: pd.DataFrame,
    Row: int,
    Col: int,
    StandardFieldID: int,
    ChannelID: int,
    TimepointID: int,
) -> PhysicalPixelSizes:
    """Determines the z,y,x resolution from the metadata.

    Parameters
    ----------
    image_metadata
        Metadata dataframe such as that generated by
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    Row
        Plate row
    Col
        Plate column
    StandardFieldID
        Field identifier (top-left to bottom-right,
        column-wise increasing first)
    ChannelID
        Channel identifier
    TimepointID
        Timepoint identifier

    Returns
    -------
    PhysicalPixelSizes
        AICSImage object for containing pixel dimensions
    """

    # z resolution is not directly provided in the image metadata,
    # derive this from the full z-range divided by number of planes
    query = (
        "Row=="
        + str(Row)
        + " & Col=="
        + str(Col)
        + " & StandardFieldID=="
        + str(StandardFieldID)
        + " & ChannelID=="
        + str(ChannelID)
        + " & TimepointID=="
        + str(TimepointID)
    )

    xyz = image_metadata.query(query)[["ImageResolutionX", "ImageResolutionY", "AbsPositionZ"]].drop_duplicates()

    return PhysicalPixelSizes(
        Z=(xyz.AbsPositionZ.max() - xyz.AbsPositionZ.min()) / (xyz.shape[0] - 1),
        Y=xyz.ImageResolutionY.iloc[0],
        X=xyz.ImageResolutionX.iloc[0],
    )


def _remove_TCZ_filename(pattern: Pattern, filename: str, mip: bool = False):
    """Modify operetta filename for Z-projection.

    Restructures a operetta tiff to remove reference to Time, Channel, Z.

    Parameters
    ----------
    pattern
        Compiled operetta regular expression
    filename
        filename to change
    mip
        Whether or not the metadata represents maximum-intensity
        projections

    Returns
    -------
    str
        Filename with TCZ information removed
    """
    m = pattern.match(filename)
    if m is not None:
        new_filename = "r" + m.group("Row") + "c" + m.group("Col") + "f" + m.group("FieldID") + "-" + m.group("End")
        if mip:
            new_filename += "-mip.ome.tiff"
        else:
            new_filename += ".ome.tiff"
        return new_filename


def _aggregate_TCZ_metadata(image_metadata: pd.DataFrame, mip: bool = False) -> pd.DataFrame:
    """Modify operetta metadata for Z-projection.

    Removes Time, Channel, Z-plane information from a dataframe. Absolute
    imaging time information is averaged across all acquisitions for a given
    time-point.

    Parameters
    ----------
    image_metadata
        Metadata dataframe such as that generated by
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    mip
        Whether or not the metadata represents maximum-intensity
        projections

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = number of imaging sites (fields-of-view))
    """
    # get average absolute time for each unique imaging site
    mean_acq_time = image_metadata.groupby(["Row", "Col", "FieldID", "TimepointID"])["AbsTime"].mean()
    n_planes = image_metadata.groupby(["Row", "Col", "FieldID", "TimepointID"])["id"].count()
    n_planes.name = "NPlanes"
    n_channels = image_metadata.groupby(["Row", "Col", "FieldID", "TimepointID"])["ChannelID"].max()
    n_channels.name = "NChannels"

    # drop columns that don't make sense after merging multiple files
    image_metadata_no_z = image_metadata.query("PlaneID==1 & ChannelID==1").drop(
        labels=[
            "ChannelID",
            "ChannelName",
            "ChannelType",
            "MainExcitationWavelength",
            "MainEmissionWavelength",
            "AbsTime",
            "AbsPositionZ",
            "PositionZ",
            "PlaneID",
        ],
        axis=1,
    )

    # remove T, C, Z information from the filename
    image_metadata_no_z["URL"] = image_metadata_no_z["URL"].apply(
        lambda x: _remove_TCZ_filename(operetta_regex_filename_pattern, x, mip)
    )

    # merge all
    image_metadata_no_z = (
        image_metadata_no_z.merge(right=mean_acq_time, on=["Row", "Col", "FieldID", "TimepointID"], how="left")
        .merge(right=n_channels, on=["Row", "Col", "FieldID", "TimepointID"], how="left")
        .merge(right=n_planes, on=["Row", "Col", "FieldID", "TimepointID"], how="left")
    )
    return image_metadata_no_z


def operetta_to_ome_tiff(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    metadata_file: str,
    n_batches: int = 1,
    batch_id: int = 0,
    save_metadata_files: bool = True,
    mip: bool = False,
    keep_stacks: bool = False,
) -> None:
    """Convert individual TIFFs from Operetta to standard OME-TIFFs.

    Reads a metadata file and loads individual TIFF files exported from the
    Perkin-Elmer Operetta. These are combined into multi-channel, multi-z-plane
    stacks and saved as OME-TIFFs.

    Parameters
    ----------
    in_path
        path to the "Images" directory (including "Images")
    out_path
        path where the OME-TIFFs should be saved
    metadata_file
        name of the xml metadata file inside "Images"
    n_batches
        number of batches into which the processing should be split.
    batch_id
        current batch to process
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    mip
        whether to save maximum-intensity-projections
    """

    if (not mip) and (not keep_stacks):
        logger.error("Neither mip nor keep_stacks are true. Images will not be saved.")

    # setup paths
    in_path = Path(in_path)
    out_path = Path(out_path)
    metadata_path = in_path / metadata_file
    if mip:
        out_path_mip = out_path.parent / (str(out_path.stem) + "-MIP")

    if in_path == out_path:
        logger.error("Input and output paths are the same.")
        os._exit(1)

    plate_metadata_file = in_path / "plate_metadata.csv" if save_metadata_files else None
    image_metadata_file = in_path / "image_metadata.csv" if save_metadata_files else None

    # make output directories
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    if mip and not out_path_mip.exists():
        out_path_mip.mkdir(parents=True, exist_ok=True)

    # get metadata dataframes
    get_plate_metadata(metadata_path, plate_metadata_file)
    image_metadata = get_image_metadata(metadata_path, image_metadata_file)

    # generate new metadata dataframes
    # (remove Z and T information from metadata)
    image_metadata_ome_tiff = _aggregate_TCZ_metadata(image_metadata)

    # save metadata files (for the first batch only)
    if save_metadata_files and int(batch_id) == 0:
        image_metadata_ome_tiff.to_csv(Path(out_path) / "image_metadata.csv", index=False)
        image_metadata_ome_tiff.to_pickle(Path(out_path) / "image_metadata.pkl")

    if mip:
        image_metadata_mip_ome_tiff = _aggregate_TCZ_metadata(image_metadata, mip=True)
        if save_metadata_files and int(batch_id) == 0:
            image_metadata_mip_ome_tiff.to_csv(Path(out_path_mip) / "image_metadata.csv", index=False)
            image_metadata_mip_ome_tiff.to_pickle(Path(out_path_mip) / "image_metadata.pkl")

    # subset metadata to the current batch only
    image_metadata = _get_metadata_batch(image_metadata=image_metadata, batch_id=batch_id, n_batches=n_batches)

    # get unique sites
    timepoints = image_metadata[["TimepointID"]].drop_duplicates()
    sites = image_metadata[["Row", "Col", "StandardFieldID"]].drop_duplicates()
    channels = image_metadata[["ChannelID"]].drop_duplicates()

    # get channel names
    channel_names = (
        image_metadata[["ChannelID", "ChannelName"]]
        .drop_duplicates()
        .sort_values(by=["ChannelID"])
        .ChannelName.tolist()
    )

    # iterate over sites, combining time-points, channels and z-planes
    # into numpy arrays to be written to files as OME-TIFFs
    for _site_index, site in sites.iterrows():
        multi_timepoint_img = []
        multi_timepoint_img_mip = []

        for _timepoint_index, timepoint in timepoints.iterrows():
            multi_channel_img = []
            multi_channel_img_mip = []

            for _channel_index, channel in channels.iterrows():
                imgs = _read_images_single_site(
                    in_path,
                    image_metadata,
                    site.Row,
                    site.Col,
                    site.StandardFieldID,
                    channel.ChannelID,
                    timepoint.TimepointID,
                )

                # combine z-planes (axis 2)
                z_stack = np.concatenate([img.data for img in imgs], axis=2)
                multi_channel_img.append(z_stack)
                if mip:
                    multi_channel_img_mip.append(z_stack.max(axis=2, keepdims=True))

            # combine channels (axis 1)
            multi_timepoint_img.append(np.concatenate(multi_channel_img, axis=1))
            if mip:
                multi_timepoint_img_mip.append(np.concatenate(multi_channel_img_mip, axis=1))

        # combine timepoints (axis 0)
        multi_timepoint_img = np.concatenate(multi_timepoint_img, axis=0)
        if mip:
            multi_timepoint_img_mip = np.concatenate(multi_timepoint_img_mip, axis=0)

        # get voxel dimensions
        voxel_dimensions = _get_zyx_resolution(image_metadata, site.Row, site.Col, site.StandardFieldID, 1, 0)

        # get new filenames from restructured metadata dataframe
        out_file_path = Path(out_path) / (
            image_metadata_ome_tiff.query(
                "Row=="
                + str(site.Row)
                + " & Col=="
                + str(site.Col)
                + " & StandardFieldID=="
                + str(site.StandardFieldID)
            ).URL.iloc[0]
        )

        if mip:
            out_file_path_mip = Path(out_path_mip) / (
                image_metadata_mip_ome_tiff.query(
                    "Row=="
                    + str(site.Row)
                    + " & Col=="
                    + str(site.Col)
                    + " & StandardFieldID=="
                    + str(site.StandardFieldID)
                ).URL.iloc[0]
            )

        # write to OME TIFF (metadata provided is written to TIFF file)
        if keep_stacks:
            OmeTiffWriter.save(
                data=multi_timepoint_img,
                uri=out_file_path,
                dim_order="TCZYX",
                channel_names=channel_names,
                physical_pixel_sizes=voxel_dimensions,
                parser="lxml",
            )

        # write MIP to OME TIFF
        if mip:
            OmeTiffWriter.save(
                data=multi_timepoint_img_mip,
                uri=out_file_path_mip,
                dim_order="TCZYX",
                channel_names=channel_names,
                physical_pixel_sizes=voxel_dimensions,
                parser="lxml",
            )
    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="operetta_to_ome_tiff")

    parser.add_argument("-i", "--in_path", help="directory containing the input files", required=True)

    parser.add_argument("-o", "--out_path", help="directory to write the output files", required=True)

    parser.add_argument(
        "--image_format",
        default="TIFF",
        help="output format for images (currently only TIFF implemented)",
    )

    parser.add_argument(
        "-f",
        "--metadata_file",
        default="Index.idx.xml",
        help="name of the metadata file",
        required=True,
    )

    parser.add_argument(
        "--batch",
        nargs=2,
        default=[1, 0],
        help="""
            If files are processed as batches, provide the number of
            batches and the current batch to be processed. Batches
            refer to the number of sites (fields-of-view) and batch
            numbers start at 0.
        """,
    )

    parser.add_argument(
        "-s",
        "--save_metadata_files",
        default=False,
        action="store_true",
        help="flag to indicate that metadata files should be saved",
    )

    parser.add_argument(
        "-m",
        "--mip",
        default=False,
        action="store_true",
        help="whether to save maximum intensity projections",
    )

    parser.add_argument(
        "--keep_stacks",
        default=False,
        action="store_true",
        help="Whether to save image stacks (all z-planes)? (default = False)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (e.g. -vvv)",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    operetta_to_ome_tiff(
        in_path=args.in_path,
        out_path=args.out_path,
        metadata_file=args.metadata_file,
        n_batches=args.batch[0],
        batch_id=args.batch[1],
        save_metadata_files=args.save_metadata_files,
        mip=args.mip,
        keep_stacks=args.keep_stacks,
    )
