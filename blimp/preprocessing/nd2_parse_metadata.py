"""Extract and parse metadata from Nikon nd2 files."""
from typing import Union
from pathlib import Path
import os
import re
import json
import logging
import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

image_metadata_dtypes = {
    "n_pixels_y": str,
    "n_pixels_x": str,
    "objective_name": str,
    "pixel_size_microns": float,
    "stage_x_abs": float,
    "stage_y_abs": float,
    "stage_z_abs": float,
    "acquisition_time_rel": float,
    "stage_z_id": np.uint16,
    "field_id": np.uint16,
}


def split_acquisition_metadata_planes(l: list) -> list:
    """Splits a list of metadata fields into channel-specific metadata sublists
    by the occurrence of "Plane".

    Parameters
    ----------
    l
        List of metadata fields

    Returns
    -------
    z
        List of metadata lists for each plane
    """
    # https://stackoverflow.com/questions/69832116/split-a-list-into-sublists-based-on-the-value-of-an-element
    x = [i for i, s in enumerate(l) if re.search("^Plane", s.lstrip())]
    y = x[1:] + [len(l)]
    z = [l[i:j] for i, j in zip(x, y)]
    return z


def parse_additional_metadata(acq_metadata: dict) -> list:
    """Extracts relevant info from a metadata dict from
    ND2Reader.parser._raw_metadata.image_text_info.

    Parameters
    ----------
    acq_metadata
        metadata from ND2Reader.parser._raw_metadata.image_text_info

    Returns
    -------
    metadata
        list of additional metadata as individual strings for each channel
    """
    logger.debug("Parsing additional metadata")
    metadata_planes = split_acquisition_metadata_planes(acq_metadata["TextInfoItem_5"].split("\r\n"))
    metadata = ["\\n".join(plane) for plane in metadata_planes]
    metadata = [s.replace(",", ";") for s in metadata]

    return metadata


def get_start_time_abs(raw_metadata: dict, acq_metadata: dict) -> datetime.datetime:
    """Finds the absolute start time from the metadata.

    Parameters
    ----------
    raw_metadata
        metadata from ND2Reader.parser._raw_metadata.__dict__
    acq_metadata
        metadata from ND2Reader.parser._raw_metadata.image_text_info

    Returns
    -------
    start_time_abs
        absolute start time
    """
    start_time_abs = raw_metadata["date"]
    if start_time_abs is None:
        logger.info("Absolute start time not found in ND2Reader.parser._raw_metadata.__dict__")
    else:
        logger.info(f"Absolute start time {start_time_abs} found in ND2Reader.parser._raw_metadata.__dict__")

    if start_time_abs is None:
        logger.info("Checking ND2Reader.parser._raw_metadata.image_text_info for absolute start time")
        start_time_abs = acq_metadata["TextInfoItem_9"]
        start_time_abs = datetime.datetime.strptime(start_time_abs, "%d/%m/%Y  %I:%M:%S %p")

    if start_time_abs is None:
        logger.warn("Absolute start time not found. Only relative time information available")

    return start_time_abs


def get_standard_field_id_mapping(df: pd.DataFrame, y_direction: str = "down") -> pd.DataFrame:
    """Convert field ids to standard field ids, with top-left to bottom-right
    ordering.

    Parameters
    ----------
    df
        metadata dataframe containing coordinates are field_ids
    y_direction
        direction of increasing (stage) y-coordinates (possible
        values are "up" and "down")

    Returns
    -------
    df
        dataframe containing a mapping of field_ids to standard_field_ids
    """
    logger.debug("Getting standard_field_id from stage coordinates")
    logger.debug(f"Y-axis direction : {y_direction}")
    if y_direction not in {"up", "down"}:
        logger.error(f'Y-axis direction : {y_direction}, only "up" or "down" are possible')
        os._exit(1)

    df = df[["field_id", "stage_x_abs", "stage_y_abs"]].groupby("field_id").mean()
    df[["stage_x_abs", "stage_y_abs"]] = df[["stage_x_abs", "stage_y_abs"]].round()
    df["XYCoordinates"] = df[["stage_x_abs", "stage_y_abs"]].apply(tuple, axis=1)
    df = df.reset_index()

    # Number fields from top-left to bottom-right (increase x first)
    if y_direction == "up":
        unique_int_coords_sorted = sorted(list(set(df["XYCoordinates"])), key=lambda k: [-k[1], k[0]])
    elif y_direction == "down":
        unique_int_coords_sorted = sorted(list(set(df["XYCoordinates"])), key=lambda k: [k[1], k[0]])
    coord_index = dict(
        zip(
            unique_int_coords_sorted,
            ["%0d" % i for i in range(1, len(unique_int_coords_sorted) + 1)],
        )
    )

    # keep this as StandardFieldID
    df["standard_field_id"] = df["XYCoordinates"].map(coord_index)

    return df[["field_id", "standard_field_id"]]


def nd2_extract_metadata_and_save(
    in_file_path: Union[str, Path],
    out_path: Union[str, Path],
    acquisition_increment_order: str = "TFZ",
    mip: bool = False,
    y_direction: str = "down",
) -> pd.DataFrame:
    """Extract metadata from .nd2 file using ND2Reader, parse and save metadata
    files.

    Parameters
    ----------
    in_file_path
        Full path to the .nd2 image file
    out_path
        Full path to the folder for OME-TIFFs
    acquisition_increment_order
        order in which field-of-view (F), time-point (T),
        and Z-position (Z) were incremented during acquisition
        (written from outer-most to inner-most loop)
        Note: only 'TFZ' currently supported.
    mip
        Should metadata be processed to reflect
        maximum-intensity projection?
    y_direction
        direction of increasing (stage) y-coordinates (possible
        values are "up" and "down")

    Returns
    -------
    Dataframe containing the metadata written to file
    """
    from nd2reader import ND2Reader

    logger.info(f"Acquisition_increment_order specified as {acquisition_increment_order}")
    if acquisition_increment_order != "TFZ":
        logger.error(
            """
        acquisition_increment_order is {}.
        Only 'TFZ' is currently supported.
        Please implement others if necessary.
        """.format(
                acquisition_increment_order
            )
        )
        os._exit(1)

    nd2_file = ND2Reader(str(in_file_path))
    acquisition_times = [t for t in nd2_file.parser._raw_metadata.acquisition_times]
    common_metadata = nd2_file.parser._raw_metadata.image_text_info[b"SLxImageTextInfo"]
    common_metadata = {key.decode(): val.decode() for key, val in common_metadata.items()}

    # save 'SLxImageTextInfo' as JSON (as backup)
    logger.debug(f"Writing JSON of ND2 metadata for file {in_file_path}")
    json_file_path = Path(out_path) / Path(Path(in_file_path).stem + ".json")
    with open(json_file_path, "w") as outfile:
        json.dump(common_metadata, outfile)

    # parse metadata
    additional_metadata = parse_additional_metadata(common_metadata)
    additional_metadata_df = pd.DataFrame(additional_metadata).T
    additional_metadata_df.columns = [
        "metadata_string_acquisition_" + str(i) for i in range(0, len(additional_metadata))
    ]

    # metadata parsed by nd2reader
    metadata_dict = nd2_file.parser._raw_metadata.__dict__

    # combine into dataframe
    metadata_df = pd.DataFrame(
        data={
            "n_pixels_y": metadata_dict["height"],
            "n_pixels_x": metadata_dict["width"],
            "objective_name": common_metadata["TextInfoItem_13"],
            "pixel_size_microns": metadata_dict["pixel_microns"],
            "stage_x_abs": nd2_file.parser._raw_metadata.x_data,
            "stage_y_abs": nd2_file.parser._raw_metadata.y_data,
            "stage_z_abs": nd2_file.parser._raw_metadata.z_data,
            "acquisition_time_rel": acquisition_times,
            "stage_z_id": list(metadata_dict["z_levels"]) * (nd2_file.sizes["t"] * nd2_file.sizes["v"]),
            "field_id": list(np.repeat(range(1, 1 + nd2_file.sizes["v"]), nd2_file.sizes["z"])) * nd2_file.sizes["t"],
            "timepoint_id": list(
                np.repeat(
                    range(nd2_file.sizes["t"]),
                    nd2_file.sizes["z"] * nd2_file.sizes["v"],
                )
            ),
        }
    )

    metadata_df["filename_ome_tiff"] = [
        Path(in_file_path).stem + "_" + str(f).zfill(4) + ".ome.tiff" for f in metadata_df["field_id"]
    ]

    # enforce types
    metadata_df = metadata_df.astype(image_metadata_dtypes)

    # remove z positions and average over z-planes for MIP metadata
    if mip:
        logger.debug(f"Aggregating metadata for MIP: {Path(out_path) / Path(Path(in_file_path).stem)}")
        aggregated = metadata_df.groupby(["field_id", "timepoint_id"])[
            ["acquisition_time_rel", "stage_y_abs", "stage_x_abs"]
        ].mean()
        metadata_df = metadata_df.drop(
            [
                "stage_y_abs",
                "stage_x_abs",
                "stage_z_abs",
                "stage_z_id",
                "acquisition_time_rel",
            ],
            axis=1,
        ).drop_duplicates()
        metadata_df = metadata_df.merge(aggregated, how="left", on=["field_id", "timepoint_id"])
        metadata_df["stage_z_n"] = len(metadata_dict["z_levels"])

    # generate absolute time column
    start_time_abs = get_start_time_abs(metadata_dict, common_metadata)
    if start_time_abs is not None:
        metadata_df["acquisition_time_abs"] = [
            start_time_abs + datetime.timedelta(seconds=x) for x in metadata_df["acquisition_time_rel"]
        ]

    # standardise field id (top-left to bottom-right)
    standard_field_id_mapping = get_standard_field_id_mapping(metadata_df, y_direction)
    metadata_df = pd.merge(metadata_df, standard_field_id_mapping, on="field_id", how="left")

    # add additional metadata as columns
    metadata_df = pd.merge(metadata_df, additional_metadata_df, how="cross")

    # write metadata to file
    with Path(out_path) / Path(Path(in_file_path).stem + "_metadata.csv") as out_file_path:
        metadata_df.to_csv(out_file_path, index=False)
    with Path(out_path) / Path(Path(in_file_path).stem + "_metadata.pkl") as out_file_path:
        metadata_df.to_pickle(out_file_path)

    return metadata_df
