"""Convert Nikon nd2 files to OME-NGFF (OME-Zarr) format.

Each nd2 file is treated as a single well containing several stage-position
fields of view, which are stitched into one contiguous mosaic image per well
using nominal grid placement from stage coordinates (no sub-pixel
registration). Wells are written into one shared OME-Zarr plate store per
plate (HCS layout), rather than one independent store per well, to avoid the
"many small files" problem that a store-per-well layout would create.

Plate/well registration and NGFF metadata construction live in
``blimp.ome_ngff`` (shared across every source format); this module supplies
only what's specific to reading an nd2 file directly -- stage positions,
voxel size, and channel metadata, via the ``nd2`` package's own structured
API (pinned -- see ``requirements.txt``).
"""
from typing import List, Union, Optional
from pathlib import Path
import logging

from ngio import open_ome_zarr_plate
from bioio import BioImage
import nd2
import numpy as np

from blimp.log import configure_logging
from blimp.ome_ngff import locate_well, NUM_PYRAMID_LEVELS, ensure_plate_exists
from blimp.ome_ngff.plate import _write_well_image
from blimp.ome_ngff.layout import (
    FieldLayout,
    _parse_well_name,
    _cluster_grid_index,
    _exact_pixel_offset,
)
from blimp.preprocessing.nd2_to_ome_tiff import _get_list_of_files_current_batch

logger = logging.getLogger(__name__)


def get_field_layout(
    nd2_path: Union[str, Path],
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
) -> FieldLayout:
    """Compute the stitching layout for an nd2 file's fields of view.

    Reads stage positions, voxel size, and channel metadata directly from
    the nd2 file via the ``nd2`` package's own structured API, rather than
    via the older ``nd2reader``-based metadata pipeline in
    ``nd2_parse_metadata.py``, for more robust cross-version compatibility.

    Parameters
    ----------
    nd2_path
        Full path to the .nd2 image file.
    y_direction
        Direction of increasing (stage) y-coordinates (possible values are
        "up" and "down"), matching the convention already established in
        ``nd2_parse_metadata.py::get_standard_field_id_mapping``.
    x_direction
        Direction of increasing (stage) x-coordinates relative to the image
        ("left" or "right"). Unlike ``y_direction``, this has no precedent
        elsewhere in blimp -- ``nd2_to_ome_tiff.py``/``nd2_parse_metadata.py``
        never needed it, since they number fields rather than placing them on
        a stitched canvas. Added after a real microscope was found to need
        it: some stage/camera combinations report x independently of y, so a
        correct y_direction does not guarantee x needs no equivalent flip.
        Defaults to "left", the most commonly used setup in this lab.
        Override per-instrument if a different microscope needs "right"
        instead.
    placement
        "grid" (default) clusters fields into rows/columns first (see
        :func:`blimp.ome_ngff.layout._cluster_grid_index`) and places each at
        an exact multiple of the tile size -- looks better when fields are
        meant to sit flush with no deliberate overlap, since it discards the
        small (a few percent of tile size) stage jitter/axis cross-talk an
        unsnapped offset would otherwise bake into the canvas. "exact" uses
        the raw stage-position offset directly instead (both modes read the
        same recorded stage positions -- this is the only difference), which
        is more faithful to the actual recorded positions -- prefer it if
        fields have genuine deliberate overlap, or nominal positions are not
        expected to fall on a regular grid.

    Returns
    -------
    FieldLayout
    """
    if y_direction not in {"up", "down"}:
        raise ValueError(f'y_direction = {y_direction}, only "up" or "down" are possible')
    if x_direction not in {"left", "right"}:
        raise ValueError(f'x_direction = {x_direction}, only "left" or "right" are possible')
    if placement not in {"grid", "exact"}:
        raise ValueError(f'placement = {placement}, only "grid" or "exact" are possible')

    with nd2.ND2File(str(nd2_path)) as f:
        voxel_size = f.voxel_size()
        sizes = f.sizes
        tile_shape = (
            sizes.get("T", 1),
            sizes.get("C", 1),
            sizes.get("Z", 1),
            sizes["Y"],
            sizes["X"],
        )

        positions = None
        for loop in f.experiment:
            if type(loop).__name__ == "XYPosLoop":
                positions = loop.parameters.points
                break

        if positions is None:
            # single-position acquisition: one field at a nominal (0, 0) offset
            position_names: List[Optional[str]] = [None]
            offsets = [(0, 0)]
        else:
            position_names = [p.name for p in positions]
            stage_x = np.array([p.stagePositionUm.x for p in positions])
            stage_y = np.array([p.stagePositionUm.y for p in positions])

            if placement == "grid":
                tile_extent_x = tile_shape[4] * voxel_size.x
                tile_extent_y = tile_shape[3] * voxel_size.y
                col_idx = _cluster_grid_index(stage_x, tile_extent_x)
                row_idx = _cluster_grid_index(stage_y, tile_extent_y)
                if x_direction == "left":
                    col_idx = col_idx.max() - col_idx
                if y_direction == "up":
                    row_idx = row_idx.max() - row_idx
                offset_x_px = col_idx * tile_shape[4]
                offset_y_px = row_idx * tile_shape[3]
            else:
                offset_x_px = _exact_pixel_offset(stage_x, voxel_size.x, reverse=(x_direction == "left"))
                offset_y_px = _exact_pixel_offset(stage_y, voxel_size.y, reverse=(y_direction == "up"))
            offsets = list(zip(offset_y_px.tolist(), offset_x_px.tolist()))

        row, column = _parse_well_name(nd2_path, position_names)

        canvas_y = max(y0 for y0, _ in offsets) + tile_shape[3]
        canvas_x = max(x0 for _, x0 in offsets) + tile_shape[4]
        canvas_shape = (tile_shape[0], tile_shape[1], tile_shape[2], canvas_y, canvas_x)

        channel_names = [ch.channel.name for ch in f.metadata.channels]
        channel_colors = [
            f"{ch.channel.color.r:02X}{ch.channel.color.g:02X}{ch.channel.color.b:02X}" for ch in f.metadata.channels
        ]

    return FieldLayout(
        row=row,
        column=column,
        offsets=offsets,
        tile_shape=tile_shape,
        canvas_shape=canvas_shape,
        pixel_size_x=voxel_size.x,
        pixel_size_y=voxel_size.y,
        pixel_size_z=voxel_size.z,
        channel_names=channel_names,
        channel_colors=channel_colors,
        position_names=position_names,
    )


def convert_individual_nd2_to_ome_ngff(
    in_file_path: Union[str, Path],
    plate_path: Union[str, Path],
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    num_levels: int = NUM_PYRAMID_LEVELS,
) -> None:
    """Stitch one nd2 file's fields of view into a well image in a shared
    OME-NGFF plate store.

    Parameters
    ----------
    in_file_path
        Full path to the .nd2 image file (one well).
    plate_path
        Full path to the shared plate .zarr store. Must already exist --
        see :func:`blimp.ome_ngff.ensure_plate_exists`.
    keep_stacks
        Whether to write the full z-stack image (at image path "stack").
    mip
        Whether to write a maximum-intensity-projection image (at image
        path "mip").
    y_direction
        Direction of increasing (stage) y-coordinates ("up" or "down").
    x_direction
        Direction of increasing (stage) x-coordinates ("left" or "right").
        See :func:`get_field_layout` for why this exists alongside
        ``y_direction``.
    placement
        "grid" or "exact" -- see :func:`get_field_layout`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect.
    num_levels
        Number of pyramid levels to write.
    """
    if not keep_stacks and not mip:
        logger.error("Neither keep_stacks nor mip are true. Nothing will be written.")
        return

    logger.info(f"Reading layout for {in_file_path}")
    layout = get_field_layout(in_file_path, y_direction=y_direction, x_direction=x_direction, placement=placement)

    if channel_names is None:
        channel_names = layout.channel_names
    elif isinstance(channel_names, str):
        channel_names = [channel_names]

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r+")

    images = BioImage(str(in_file_path))

    def get_tile(field_index: int) -> np.ndarray:
        images.set_scene(field_index)
        return images.get_image_data("TCZYX")

    if keep_stacks:
        _write_well_image(
            get_tile=get_tile,
            layout=layout,
            plate=plate,
            plate_path=plate_path,
            image_path="stack",
            channel_names=channel_names,
            channel_colors=layout.channel_colors,
            dtype=images.dtype,
            num_levels=num_levels,
            project_z=False,
        )

    if mip:
        _write_well_image(
            get_tile=get_tile,
            layout=layout,
            plate=plate,
            plate_path=plate_path,
            image_path="mip",
            channel_names=channel_names,
            channel_colors=layout.channel_colors,
            dtype=images.dtype,
            num_levels=num_levels,
            project_z=True,
        )


def nd2_to_ome_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    plate_name: Optional[str] = None,
    n_batches: int = 1,
    batch_id: int = 0,
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
) -> None:
    """Read a folder of nd2 files (one well each) and stitch them into a
    shared OME-NGFF plate store. Can perform batch processing.

    Note: this does not itself call :func:`blimp.ome_ngff.ensure_plate_exists`
    for every file -- it assumes the plate store already exists (call
    :func:`blimp.ome_ngff.ensure_plate_exists` once beforehand, e.g. from the
    same serial step that discovers input files and generates per-batch jobs)
    so that concurrent batches never race to create it.

    Parameters
    ----------
    in_path
        Full path to the folder of .nd2 image files.
    plate_path
        Full path to the shared plate .zarr store.
    plate_name
        Name for the plate, used only if it does not already exist (see
        :func:`blimp.ome_ngff.ensure_plate_exists`).
    n_batches
        Number of batches into which the processing should be split.
    batch_id
        Current batch to process.
    keep_stacks
        Whether to write full z-stack images.
    mip
        Whether to write maximum-intensity-projection images.
    y_direction
        Direction of increasing (stage) y-coordinates ("up" or "down").
    x_direction
        Direction of increasing (stage) x-coordinates ("left" or "right").
    placement
        "grid" or "exact" -- see :func:`get_field_layout`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect and need to be replaced.
    """
    in_path = Path(in_path)
    plate_path = Path(plate_path)

    ensure_plate_exists(plate_path, plate_name or plate_path.stem)

    filename_list = _get_list_of_files_current_batch(in_path=in_path, n_batches=n_batches, batch_id=batch_id)

    logger.info(f"Converting nd2 files to OME-NGFF: {filename_list}")
    for f in filename_list:
        in_file_path = in_path / f
        convert_individual_nd2_to_ome_ngff(
            in_file_path=in_file_path,
            plate_path=plate_path,
            keep_stacks=keep_stacks,
            mip=mip,
            y_direction=y_direction,
            x_direction=x_direction,
            placement=placement,
            channel_names=channel_names,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="nd2_to_ome_ngff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert a folder of .nd2 files to OME-NGFF")
    convert_parser.add_argument("-i", "--in_path", help="directory containing the input .nd2 files", required=True)
    convert_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    convert_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    convert_parser.add_argument(
        "--batch",
        nargs=2,
        default=[1, 0],
        metavar=("N_BATCHES", "BATCH_ID"),
        help="number of batches and the current batch to process (0-indexed)",
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
        default="down",
        help='direction of increasing (stage) y-coordinates ("up" or "down")',
    )
    convert_parser.add_argument(
        "-x",
        "--x_direction",
        default="left",
        help='direction of increasing (stage) x-coordinates ("left" or "right")',
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

    ensure_parser = subparsers.add_parser("ensure-plate", help="Idempotently create a plate store if it doesn't exist")
    ensure_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    ensure_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    ensure_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    locate_parser = subparsers.add_parser("locate-well", help="Print the on-disk path of one well")
    locate_parser.add_argument("-p", "--plate_path", help="path to the shared plate .zarr store", required=True)
    locate_parser.add_argument("-w", "--well", help="well name, e.g. C09", required=True)
    locate_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "convert":
        nd2_to_ome_ngff(
            in_path=args.in_path,
            plate_path=args.plate_path,
            plate_name=args.plate_name,
            n_batches=args.batch[0],
            batch_id=args.batch[1],
            keep_stacks=args.keep_stacks,
            mip=args.mip,
            y_direction=args.y_direction,
            x_direction=args.x_direction,
            placement=args.placement,
            channel_names=args.channel_names,
        )
    elif args.command == "ensure-plate":
        ensure_plate_exists(args.plate_path, args.plate_name or Path(args.plate_path).stem)
    elif args.command == "locate-well":
        locate_well(args.plate_path, args.well)
