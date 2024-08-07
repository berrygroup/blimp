"""Convert Nikon nd2 files to open microscopy environment OME-TIFF format."""
from glob import glob
from typing import List, Union
from pathlib import Path
import os
import logging

from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
import numpy as np

from blimp.log import configure_logging
from blimp.preprocessing.nd2_parse_metadata import nd2_extract_metadata_and_save

logger = logging.getLogger(__name__)


def convert_individual_nd2_to_ome_tiff(
    in_file_path: Union[str, Path],
    out_path: Union[str, Path, None],
    out_path_mip: Union[str, Path, None] = None,
    channel_names: Union[str, List[str], None] = None,
):
    """Reads an nd2 file and writes a set of image files corresponding to
    single imaging sites (field of view).

    Parameters
    ----------
    in_file_path
        Full path to the .nd2 image file
    out_path
        Full path to the folder for OME-TIFFs
    out_path_mip
        Full path to the folder z-projected OME-TIFFs
    channel_names
        List of channel names in case those found in the
        image metadata are incorrect
    Returns
    -------
    """

    logger.info(f"Reading individual ND2 file {in_file_path}")
    images = AICSImage(str(in_file_path))

    if channel_names is None:
        channel_names = images.channel_names
        logger.debug(f"Using channel names from image file {channel_names}.")
    elif isinstance(channel_names, str):
        channel_names = [channel_names]
    elif isinstance(channel_names, list):
        for channel_name in channel_names:
            if not isinstance(channel_name, str):
                raise ValueError("Channel names must be strings.")
        logger.debug(f"Using channel names from input {channel_names}.")
    else:
        ValueError("Unknown error in channel names.")

    for s, scene in enumerate(images.scenes):
        if out_path is not None or out_path_mip is not None:
            images.set_scene(s)
            image_data = images.get_image_data("TCZYX")

        if out_path is not None:
            out_file_path = Path(out_path) / Path(Path(in_file_path).stem + "_" + str(s + 1).zfill(4) + ".ome.tiff")

            OmeTiffWriter.save(
                data=image_data,
                uri=out_file_path,
                dim_order="TCZYX",
                channel_names=channel_names,
                physical_pixel_sizes=images.physical_pixel_sizes,
                parser="lxml",
            )

        if out_path_mip is not None:
            out_file_path_mip = Path(out_path_mip) / Path(
                Path(in_file_path).stem + "_" + str(s + 1).zfill(4) + ".ome.tiff"
            )

            logger.debug(f"Writing OME-TIFF MIP, field-of-view #{s}")
            OmeTiffWriter.save(
                data=np.max(image_data, axis=2, keepdims=True),
                uri=out_file_path_mip,
                dim_order="TCZYX",
                channel_names=channel_names,
                physical_pixel_sizes=images.physical_pixel_sizes,
                parser="lxml",
            )

    return


def convert_individual_nd2_to_ome_tiff_nd2_reader(
    in_file_path: Union[str, Path],
    out_path: Union[str, Path, None],
    out_path_mip: Union[str, Path, None] = None,
):
    """Reads an nd2 file and writes a set of image files corresponding to
    single imaging sites (field of view).

    Parameters
    ----------
    in_file_path
        Full path to the .nd2 image file
    out_path
        Full path to the folder for OME-TIFFs
    out_path_mip
        Full path to the folder z-projected OME-TIFFs
    Returns
    -------
    """
    from nd2reader import ND2Reader

    logger.info(f"Reading individual ND2 file {in_file_path}")
    images = ND2Reader(str(in_file_path))

    images.sizes["v"]

    images.bundle_axes = "tczyx"
    images.iter_axes = "v"

    for i, img in enumerate(images):
        voxel_dimensions = _get_zyx_resolution(img.metadata)

        if out_path is not None:
            out_file_path = Path(out_path) / Path(Path(in_file_path).stem + "_" + str(i + 1).zfill(4) + ".ome.tiff")

            logger.debug(f"Writing OME-TIFF, field-of-view #{i}")
            OmeTiffWriter.save(
                data=img,
                uri=out_file_path,
                dim_order="TCZYX",
                channel_names=img.metadata["channels"],
                physical_pixel_sizes=voxel_dimensions,
                parser="lxml",
            )

        if out_path_mip is not None:
            out_file_path_mip = Path(out_path_mip) / Path(
                Path(in_file_path).stem + "_" + str(i + 1).zfill(4) + ".ome.tiff"
            )

            logger.debug(f"Writing OME-TIFF MIP, field-of-view #{i}")
            OmeTiffWriter.save(
                data=np.max(img, axis=2, keepdims=True),
                uri=out_file_path_mip,
                dim_order="TCZYX",
                channel_names=img.metadata["channels"],
                physical_pixel_sizes=voxel_dimensions,
                parser="lxml",
            )

    return out_file_path


def _get_zyx_resolution(image_metadata: dict) -> PhysicalPixelSizes:
    """Determines the z,y,x resolution from the metadata.

    Parameters
    ----------
    image_metadata
        Metadata dict generated by ND2Reader

    Returns
    -------
    PhysicalPixelSizes
        AICSImage object for containing pixel dimensions
    """
    logger.debug("Getting voxel dimensions")
    image_metadata["pixel_microns"]
    n_z = 1 + max([i for i in image_metadata["z_levels"]])
    return PhysicalPixelSizes(
        Z=(max(image_metadata["z_coordinates"][0:n_z]) - min(image_metadata["z_coordinates"][0:n_z])) / (n_z - 1),
        Y=image_metadata["pixel_microns"],
        X=image_metadata["pixel_microns"],
    )


def _get_list_of_files_current_batch(in_path: Union[str, Path], batch_id: int, n_batches: int) -> list:
    """Get a list of files to process in batch mode.

    Parameters
    ----------
    in_path
        Full path to the folder of .nd2 image files
    batch_id
        0-indexed batch id
    n_batches
        How many batches the files should be processed in

    Returns
    -------
    List[str]
        list of files to process in this batch
    """

    batch_id = int(batch_id)
    n_batches = int(n_batches)

    # get reproducible list of nd2 files in 'in_path'
    in_path = Path(in_path)
    filepaths = glob(str(in_path / "*.nd2"))
    # FIXME: this function works only for absolute paths!
    # filepaths = [Path(f).name for f in filepaths]
    filepaths.sort()
    logger.debug(f"{len(filepaths)} files found:")
    for i, f in enumerate(filepaths):
        logger.debug(f"Input file {i}: {f}")

    n_files_per_batch = int(np.ceil(float(len(filepaths)) / float(n_batches)))

    logger.info(f"Splitting nd2 file list into {n_batches} batches of size {n_files_per_batch}")
    logger.info(f"Processing batch {batch_id}")

    return filepaths[batch_id * n_files_per_batch : (batch_id + 1) * n_files_per_batch]


def nd2_to_ome_tiff(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    n_batches: int = 1,
    batch_id: int = 0,
    mip: bool = False,
    keep_stacks: bool = False,
    y_direction: str = "down",
    channel_names: Union[str, List[str], None] = None,
) -> None:
    """Reads an folder of nd2 files and converts to OME-TIFFs. Can perform
    batch processing.

    Parameters
    ----------
    in_path
        Full path to the folder of .nd2 image files
    out_path
        Full path to the folder for OME-TIFFs
    n_batches
        number of batches into which the processing should be split.
    batch_id
        current batch to process
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    y_direction
        direction of increasing (stage) y-coordinates (possible
        values are "up" and "down")
    channel_names
        List of channel names in case those found in the
        image metadata are incorrect and need to be replaced

    Returns
    -------
    """

    # setup paths
    in_path = Path(in_path)
    out_path = Path(out_path)
    if mip:
        out_path_mip = out_path.parent / (str(out_path.stem) + "-MIP")
    else:
        out_path_mip = None

    if in_path == out_path:
        logger.error("Input and output paths are the same.")
        os._exit(1)

    # make output directories
    if not out_path.exists():
        logger.info(f"Created output directory: {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
    if mip and isinstance(out_path_mip, Path):
        if not out_path_mip.exists():
            logger.info(f"Created output directory: {out_path_mip}")
            out_path_mip.mkdir(parents=True, exist_ok=True)

    # get list of files to process
    filename_list = _get_list_of_files_current_batch(in_path=in_path, n_batches=n_batches, batch_id=batch_id)

    # if keep_stacks is False, out_path to None
    if keep_stacks:
        out_path_stack = out_path
    else:
        out_path_stack = None

    logger.info(f"Converting nd2 files: {filename_list}")
    for f in filename_list:
        in_file_path = in_path / f
        convert_individual_nd2_to_ome_tiff(
            in_file_path=in_file_path,
            out_path=out_path_stack,
            out_path_mip=out_path_mip,
            channel_names=channel_names,
        )

        # save metadata
        if out_path_stack is not None:
            logger.info(f"Saving metadata for {in_file_path} in {out_path_stack}")
            nd2_metadata = nd2_extract_metadata_and_save(
                in_file_path=in_file_path, out_path=out_path_stack, y_direction=y_direction
            )

        # save mip metadata
        if out_path_mip is not None:
            logger.info(f"Saving MIP metadata for {in_file_path} in {out_path_mip}")
            nd2_metadata = nd2_extract_metadata_and_save(
                in_file_path=in_file_path,
                out_path=out_path_mip,
                mip=True,
                y_direction=y_direction,
            )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="nd2_to_ome_tiff")

    parser.add_argument("-i", "--in_path", help="directory containing the input files", required=True)

    parser.add_argument("-o", "--out_path", help="directory to write the output files", required=True)

    parser.add_argument(
        "--image_format",
        default="TIFF",
        help="output format for images (currently only TIFF implemented)",
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
        "-y",
        "--y_direction",
        default="down",
        help="""
        Microscope stages can have inconsistent y orientations
        relative to images. Standardised field identifiers are
        derived from microscope stage positions but to ensure
        the orientation of the y-axis relative to images, this
        must be specified. Default value is "down" so that
        y-coordinate values increase as the stage moves toward
        the eyepiece. Change to "up" if stiching doesn't look
        right!
        """,
    )

    parser.add_argument(
        "-c",
        "--channel_names",
        type=str,
        nargs="+",
        default=None,
        help="List of channel names",
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

    nd2_to_ome_tiff(
        in_path=args.in_path,
        out_path=args.out_path,
        n_batches=args.batch[0],
        batch_id=args.batch[1],
        mip=args.mip,
        keep_stacks=args.keep_stacks,
        y_direction=args.y_direction,
        channel_names=args.channel_names,
    )
