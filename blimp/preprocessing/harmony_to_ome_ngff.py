"""Convert Perkin-Elmer (Revvity) Harmony exports -- Operetta and Opera
Phenix -- to OME-NGFF (OME-Zarr) format: each well's fields of view are
stitched into one contiguous mosaic and written into a shared, plate-wide
OME-Zarr store (HCS layout). Plate/well registration and NGFF metadata
construction live in ``blimp.ome_ngff`` (shared across every source format).
"""
from typing import List, Tuple, Union, Optional
from pathlib import Path
import json
import logging

from ngio import open_ome_zarr_plate
from bioio import BioImage
import numpy as np
import pandas as pd

from blimp.log import configure_logging
from blimp.utils import pbs_array_directive
from blimp.ome_ngff import locate_well, NUM_PYRAMID_LEVELS, ensure_plate_exists
from blimp.preprocessing import opera_parse_metadata, operetta_parse_metadata
from blimp.ome_ngff.plate import _write_well_image
from blimp.ome_ngff.layout import FieldLayout, _field_offsets, _parse_well_name
from blimp.preprocessing.operetta_to_ome_tiff import (
    _get_zyx_resolution,
    _read_images_single_site,
)
from blimp.preprocessing.operetta_parse_metadata import _to_well_name

logger = logging.getLogger(__name__)

_PARSERS = {"opera": opera_parse_metadata, "operetta": operetta_parse_metadata}
_METADATA_FILES = {"opera": "index.xml", "operetta": "Index.idx.xml"}
_DEFAULT_CHANNEL_COLORS = ["FF0000", "00FF00", "0000FF", "FF00FF", "00FFFF", "FFFF00"]


def _check_instrument(instrument: str) -> None:
    if instrument not in _PARSERS:
        raise ValueError(f'instrument = {instrument}, only "opera" or "operetta" are possible')


def _get_metadata_path(in_path: Union[str, Path], instrument: str, metadata_file: Optional[str] = None) -> Path:
    _check_instrument(instrument)
    return Path(in_path) / (metadata_file or _METADATA_FILES[instrument])


def get_plate_size(plate_metadata: pd.DataFrame) -> str:
    """Standard plate format ("96" or "384") from Harmony plate metadata,
    defaulting to "384" if the plate is any other size."""
    n_wells = int(plate_metadata["PlateRows"].iloc[0]) * int(plate_metadata["PlateColumns"].iloc[0])
    if str(n_wells) in {"96", "384"}:
        return str(n_wells)
    logger.warning(f"Plate has {n_wells} wells, declaring a 384-well grid")
    return "384"


def ensure_harmony_plate_exists(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    instrument: str,
    metadata_file: Optional[str] = None,
):
    """Open or create the plate store for a Harmony measurement, named and
    sized from the measurement's own plate metadata (see
    :func:`blimp.ome_ngff.ensure_plate_exists`).

    Parameters
    ----------
    in_path
        path to the "images" (opera) or "Images" (operetta) directory
    plate_path
        full path to the plate .zarr store
    instrument
        "opera" or "operetta"
    metadata_file
        name of the xml metadata file inside ``in_path`` (default
        "index.xml" for opera, "Index.idx.xml" for operetta)

    Returns
    -------
    OmeZarrPlate
    """
    metadata_path = _get_metadata_path(in_path, instrument, metadata_file)
    plate_metadata = _PARSERS[instrument].get_plate_metadata(metadata_path)
    return ensure_plate_exists(
        plate_path,
        plate_name=str(plate_metadata["Name"].iloc[0]),
        plate_size=get_plate_size(plate_metadata),  # type: ignore
    )


def directions_from_orientation_matrix(orientation_matrix: str) -> Tuple[str, str]:
    """Directions of increasing stage-y and stage-x in the image, from the
    signs of the diagonal of Harmony's per-image stage-to-image
    ``OrientationMatrix`` (e.g. ``"[[0.99,0,0,-98.8],[0,-0.99,0,9.3],[0,0,1.00,-2.1]]"``).

    Returns
    -------
    (y_direction, x_direction)
        e.g. ``("up", "right")``
    """
    matrix = json.loads(orientation_matrix)
    x_direction = "right" if matrix[0][0] > 0 else "left"
    y_direction = "down" if matrix[1][1] > 0 else "up"
    return y_direction, x_direction


def _emission_color(wavelength: float) -> Optional[str]:
    """Approximate display color for an emission wavelength (nm)."""
    if pd.isna(wavelength) or wavelength <= 0:
        return None
    if wavelength < 500:
        return "0000FF"
    if wavelength < 570:
        return "00FF00"
    if wavelength < 640:
        return "FF0000"
    return "FF00FF"


def get_field_layout_from_harmony_metadata(
    well_metadata: pd.DataFrame,
    y_direction: Optional[str] = None,
    x_direction: Optional[str] = None,
    placement: str = "grid",
) -> FieldLayout:
    """Compute the stitching layout for one well's fields of view from
    parsed Harmony image metadata (as returned by
    :func:`blimp.preprocessing.opera_parse_metadata.get_image_metadata` or
    :func:`blimp.preprocessing.operetta_parse_metadata.get_image_metadata`).

    Parameters
    ----------
    well_metadata
        Image metadata for a single well.
    y_direction, x_direction
        Direction of increasing stage-y ("up" or "down") and stage-x ("left"
        or "right") in the image. ``None`` (default) derives each from the
        metadata's own ``OrientationMatrix`` (see
        :func:`directions_from_orientation_matrix`).
    placement
        "grid" or "exact" -- see
        :func:`blimp.preprocessing.nd2_to_ome_ngff.get_field_layout`.

    Returns
    -------
    FieldLayout
    """
    if well_metadata[["Row", "Col"]].drop_duplicates().shape[0] != 1:
        raise ValueError("well_metadata must contain exactly one well")

    if y_direction is None or x_direction is None:
        derived_y, derived_x = directions_from_orientation_matrix(well_metadata["OrientationMatrix"].iloc[0])
        y_direction = y_direction or derived_y
        x_direction = x_direction or derived_x
    logger.debug(f"Stage directions: y_direction={y_direction}, x_direction={x_direction}")

    fields = well_metadata.drop_duplicates(subset="FieldID").sort_values(by="FieldID")
    first = fields.iloc[0]

    tile_shape = (
        well_metadata["TimepointID"].nunique(),
        well_metadata["ChannelID"].nunique(),
        well_metadata["PlaneID"].nunique(),
        int(first.ImageSizeY),
        int(first.ImageSizeX),
    )

    pixel_size = _get_zyx_resolution(
        well_metadata,
        first.Row,
        first.Col,
        first.StandardFieldID,
        well_metadata["ChannelID"].min(),
        well_metadata["TimepointID"].min(),
    )

    # stage positions are in metres
    offsets = _field_offsets(
        fields["PositionX"].to_numpy() * 1e6,
        fields["PositionY"].to_numpy() * 1e6,
        tile_shape,
        pixel_size.X,
        pixel_size.Y,
        x_direction,
        y_direction,
        placement,
    )

    well_name = _to_well_name(int(first.Row), int(first.Col))
    row, column = _parse_well_name(well_name, [well_name])

    canvas_y = max(y0 for y0, _ in offsets) + tile_shape[3]
    canvas_x = max(x0 for _, x0 in offsets) + tile_shape[4]
    canvas_shape = (tile_shape[0], tile_shape[1], tile_shape[2], canvas_y, canvas_x)

    channels = well_metadata.drop_duplicates(subset="ChannelID").sort_values(by="ChannelID")
    channel_names = channels["ChannelName"].tolist()
    channel_colors = [
        _emission_color(wavelength) or _DEFAULT_CHANNEL_COLORS[i % len(_DEFAULT_CHANNEL_COLORS)]
        for i, wavelength in enumerate(channels["MainEmissionWavelength"])
    ]

    field_ids = [int(f) for f in fields["FieldID"]]
    position_names: List[Optional[str]] = [f"{row}{column:02d}_{field_id:04d}" for field_id in field_ids]

    return FieldLayout(
        row=row,
        column=column,
        offsets=offsets,
        tile_shape=tile_shape,
        canvas_shape=canvas_shape,
        pixel_size_x=pixel_size.X,
        pixel_size_y=pixel_size.Y,
        pixel_size_z=pixel_size.Z or 1.0,
        channel_names=channel_names,
        channel_colors=channel_colors,
        position_names=position_names,
        field_ids=field_ids,
    )


def convert_harmony_well_to_ome_ngff(
    in_path: Union[str, Path],
    well_metadata: pd.DataFrame,
    plate_path: Union[str, Path],
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: Optional[str] = None,
    x_direction: Optional[str] = None,
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    num_levels: int = NUM_PYRAMID_LEVELS,
) -> None:
    """Stitch one well's fields of view into a well image in a shared
    OME-NGFF plate store.

    Parameters
    ----------
    in_path
        path to the "images" (opera) or "Images" (operetta) directory
    well_metadata
        Image metadata for a single well (see
        :func:`get_field_layout_from_harmony_metadata`).
    plate_path
        Full path to the shared plate .zarr store. Must already exist --
        see :func:`ensure_harmony_plate_exists`.
    keep_stacks
        Whether to write the full z-stack image (at image path "stack").
    mip
        Whether to write a maximum-intensity-projection image (at image
        path "mip").
    y_direction, x_direction, placement
        See :func:`get_field_layout_from_harmony_metadata`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect.
    num_levels
        Number of pyramid levels to write.
    """
    if not keep_stacks and not mip:
        logger.error("Neither keep_stacks nor mip are true. Nothing will be written.")
        return

    layout = get_field_layout_from_harmony_metadata(
        well_metadata, y_direction=y_direction, x_direction=x_direction, placement=placement
    )

    if channel_names is None:
        channel_names = layout.channel_names
    elif isinstance(channel_names, str):
        channel_names = [channel_names]

    fields = well_metadata.drop_duplicates(subset="FieldID").sort_values(by="FieldID")
    timepoints = sorted(well_metadata["TimepointID"].unique())
    channels = sorted(well_metadata["ChannelID"].unique())

    def get_tile(field_index: int) -> np.ndarray:
        field = fields.iloc[field_index]
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [
                                img.data
                                for img in _read_images_single_site(
                                    in_path,
                                    well_metadata,
                                    field.Row,
                                    field.Col,
                                    field.StandardFieldID,
                                    channel,
                                    timepoint,
                                )
                            ],
                            axis=2,
                        )
                        for channel in channels
                    ],
                    axis=1,
                )
                for timepoint in timepoints
            ],
            axis=0,
        )

    dtype = BioImage(Path(in_path) / fields.iloc[0].URL).dtype
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r+")

    for image_path, project_z in (("stack", False), ("mip", True)):
        if (project_z and not mip) or (not project_z and not keep_stacks):
            continue
        _write_well_image(
            get_tile=get_tile,
            layout=layout,
            plate=plate,
            plate_path=plate_path,
            image_path=image_path,
            channel_names=channel_names,
            channel_colors=layout.channel_colors,
            dtype=dtype,
            num_levels=num_levels,
            project_z=project_z,
        )


def generate_pbs_script_ngff(
    template: str,
    input_dir: str,
    plate_path: str,
    log_dir: str,
    user: str,
    email: str,
    n_batches: int,
    mip: bool,
    keep_stacks: bool,
    save_metadata_files: bool,
    y_direction: Optional[str] = None,
    x_direction: Optional[str] = None,
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    conda_env: str = "berrylab-py311",
) -> str:
    """Formats an opera/operetta NGFF PBS jobscript template using input
    arguments.

    Parameters
    ----------
    template
        PBS jobscript template
    input_dir
        full path to the measurement directory (containing "images" or
        "Images")
    plate_path
        full path to the shared plate .zarr store (created up front, before
        this jobscript is written, so concurrent batch tasks never race to
        create it)
    log_dir
        full path to where output logs should be written
    user
        usename (zID on katana) for job submission and location of scripts
    email
        email address for notifications
    n_batches
        how many batches (of wells) into which processing should be split
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    y_direction, x_direction, placement
        see :func:`get_field_layout_from_harmony_metadata`
    channel_names
        List of channel names in case those found in the
        image metadata are incorrect and need to be replaced
    conda_env
        name of the conda environment to activate on the compute node

    Returns
    -------
    Template as a formatted string to be written to file
    """
    if channel_names is None:
        channel_names = ""
    else:
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        channel_names = "--channel_names " + (" ".join(channel_names))

    directions = []
    if y_direction is not None:
        directions.append(f"-y {y_direction}")
    if x_direction is not None:
        directions.append(f"-x {x_direction}")

    array_directive, batch_id_expr = pbs_array_directive(n_batches)

    return template.format(
        INPUT_DIR=input_dir,
        PLATE_PATH=plate_path,
        LOG_DIR=log_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        ARRAY_DIRECTIVE=array_directive,
        BATCH_ID_EXPR=batch_id_expr,
        MIP="--mip" if mip else "",
        KEEP_STACKS="--keep_stacks" if keep_stacks else "",
        SAVE_METADATA_FILES="--save_metadata_files" if save_metadata_files else "",
        DIRECTIONS=" ".join(directions),
        PLACEMENT=placement,
        CHANNEL_NAMES=channel_names,
        CONDA_ENV=conda_env,
    )


def _get_wells_batch(image_metadata: pd.DataFrame, n_batches: int, batch_id: int) -> List[Tuple[int, int]]:
    """Predictable subset of the plate's (Row, Col) wells for one batch."""
    wells = list(
        image_metadata[["Row", "Col"]].drop_duplicates().sort_values(by=["Row", "Col"]).itertuples(index=False)
    )
    n_wells_per_batch = int(np.ceil(len(wells) / float(n_batches)))
    logger.info(f"Splitting {len(wells)} wells into {n_batches} batches of size {n_wells_per_batch}")
    logger.info(f"Processing batch {batch_id}")
    batch = wells[int(batch_id) * n_wells_per_batch : (int(batch_id) + 1) * n_wells_per_batch]
    if len(batch) == 0:
        logger.warning("batch_id exceeds number of wells, nothing to process")
    return [(int(w.Row), int(w.Col)) for w in batch]


def harmony_to_ome_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    instrument: str,
    metadata_file: Optional[str] = None,
    n_batches: int = 1,
    batch_id: int = 0,
    save_metadata_files: bool = True,
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: Optional[str] = None,
    x_direction: Optional[str] = None,
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
) -> None:
    """Read a Harmony (opera or operetta) export and stitch each well into a
    shared OME-NGFF plate store. Can perform batch processing (by well).

    Parameters
    ----------
    in_path
        path to the "images" (opera) or "Images" (operetta) directory
    plate_path
        Full path to the shared plate .zarr store.
    instrument
        "opera" or "operetta"
    metadata_file
        name of the xml metadata file inside ``in_path`` (default
        "index.xml" for opera, "Index.idx.xml" for operetta)
    n_batches
        Number of batches into which the processing should be split.
    batch_id
        Current batch to process.
    save_metadata_files
        Whether the parsed plate and image metadata should be saved (as
        CSV, next to the plate store).
    keep_stacks
        Whether to write full z-stack images.
    mip
        Whether to write maximum-intensity-projection images.
    y_direction, x_direction, placement
        See :func:`get_field_layout_from_harmony_metadata`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect and need to be replaced.
    """
    in_path = Path(in_path)
    plate_path = Path(plate_path)
    metadata_path = _get_metadata_path(in_path, instrument, metadata_file)
    parser = _PARSERS[instrument]

    ensure_harmony_plate_exists(in_path, plate_path, instrument, metadata_file)

    write_metadata = save_metadata_files and int(batch_id) == 0
    parser.get_plate_metadata(metadata_path, plate_path.parent / "plate_metadata.csv" if write_metadata else None)
    image_metadata = parser.get_image_metadata(
        metadata_path, plate_path.parent / "image_metadata.csv" if write_metadata else None
    )

    for row, col in _get_wells_batch(image_metadata, n_batches=n_batches, batch_id=batch_id):
        logger.info(f"Converting well {_to_well_name(row, col)} to OME-NGFF")
        convert_harmony_well_to_ome_ngff(
            in_path=in_path,
            well_metadata=image_metadata.query(f"Row=={row} & Col=={col}"),
            plate_path=plate_path,
            keep_stacks=keep_stacks,
            mip=mip,
            y_direction=y_direction,
            x_direction=x_direction,
            placement=placement,
            channel_names=channel_names,
        )


def check_harmony_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    instrument: str,
    metadata_file: Optional[str] = None,
    mip: bool = False,
    keep_stacks: bool = True,
) -> set:
    """Check that every well in a Harmony export was written to the plate
    store.

    Parameters
    ----------
    in_path
        path to the "images" (opera) or "Images" (operetta) directory
    plate_path
        Full path to the plate .zarr store.
    instrument
        "opera" or "operetta"
    metadata_file
        name of the xml metadata file inside ``in_path``
    mip
        whether to check for maximum-intensity-projections
    keep_stacks
        whether to check for stacks

    Returns
    -------
    Set of missing (well name, image path) pairs, e.g. ``{("C05", "mip")}``
    """
    metadata_path = _get_metadata_path(in_path, instrument, metadata_file)
    image_metadata = _PARSERS[instrument].get_image_metadata(metadata_path)
    wells = image_metadata[["Row", "Col"]].drop_duplicates().itertuples(index=False)
    expected = {
        (_to_well_name(int(w.Row), int(w.Col)), image_path)
        for w in wells
        for image_path in (["stack"] if keep_stacks else []) + (["mip"] if mip else [])
    }

    if not Path(plate_path).exists():
        raise FileNotFoundError(f"Plate store {plate_path} not found")
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    found = set()
    for well_path in plate.wells_paths():
        row, column = well_path.split("/")
        for image_path in plate.get_well(row, column).paths():
            found.add((f"{row}{int(column):02d}", image_path))

    missing = expected.difference(found)
    if len(missing) != 0:
        logger.warning(f"The following well images are expected in {plate_path}, but were not found")
        for well_name, image_path in sorted(missing):
            logger.warning(f"Well {well_name}, image '{image_path}'")
    else:
        logger.info(f"All {len(expected)} expected well images found in {plate_path}")
        logger.info("Conversion appears complete")
    return missing


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="harmony_to_ome_ngff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert a Harmony export to OME-NGFF")
    convert_parser.add_argument(
        "-i", "--in_path", help='the "images" (opera) or "Images" (operetta) directory', required=True
    )
    convert_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    convert_parser.add_argument("--instrument", choices=["opera", "operetta"], required=True)
    convert_parser.add_argument(
        "-f", "--metadata_file", default=None, help="name of the metadata file (default depends on --instrument)"
    )
    convert_parser.add_argument(
        "--batch",
        nargs=2,
        default=[1, 0],
        metavar=("N_BATCHES", "BATCH_ID"),
        help="number of batches and the current batch to process (0-indexed), by well",
    )
    convert_parser.add_argument(
        "-s",
        "--save_metadata_files",
        default=False,
        action="store_true",
        help="flag to indicate that metadata files should be saved",
    )
    convert_parser.add_argument(
        "--keep_stacks",
        default=False,
        action="store_true",
        help="whether to write full z-stack images (default = False)",
    )
    convert_parser.add_argument(
        "-m", "--mip", default=False, action="store_true", help="whether to write maximum intensity projections"
    )
    convert_parser.add_argument(
        "-y",
        "--y_direction",
        default=None,
        choices=["up", "down"],
        help="direction of increasing (stage) y-coordinates (default: from the metadata)",
    )
    convert_parser.add_argument(
        "-x",
        "--x_direction",
        default=None,
        choices=["left", "right"],
        help="direction of increasing (stage) x-coordinates (default: from the metadata)",
    )
    convert_parser.add_argument(
        "--placement",
        default="grid",
        choices=["grid", "exact"],
        help='"grid" snaps fields to a tile grid (default); "exact" uses the raw stage offset directly',
    )
    convert_parser.add_argument(
        "-c", "--channel_names", type=str, nargs="+", default=None, help="list of channel names"
    )
    convert_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    locate_parser = subparsers.add_parser("locate-well", help="Print the on-disk path of one well")
    locate_parser.add_argument("-p", "--plate_path", help="path to the shared plate .zarr store", required=True)
    locate_parser.add_argument("-w", "--well", help="well name, e.g. C09", required=True)
    locate_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "convert":
        harmony_to_ome_ngff(
            in_path=args.in_path,
            plate_path=args.plate_path,
            instrument=args.instrument,
            metadata_file=args.metadata_file,
            n_batches=args.batch[0],
            batch_id=args.batch[1],
            save_metadata_files=args.save_metadata_files,
            keep_stacks=args.keep_stacks,
            mip=args.mip,
            y_direction=args.y_direction,
            x_direction=args.x_direction,
            placement=args.placement,
            channel_names=args.channel_names,
        )
    elif args.command == "locate-well":
        locate_well(args.plate_path, args.well)
