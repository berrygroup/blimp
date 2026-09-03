"""Shared OME-Zarr plate/well registration and image writing, used by every
OME-NGFF writer (nd2-sourced, TIFF-sourced, and future Operetta-sourced)."""
from typing import Any, List, Union, Literal, Callable
from pathlib import Path
import string
import logging

from ngio import (
    OmeZarrContainer,
    create_empty_plate,
    NgioFileExistsError,
    open_ome_zarr_plate,
    NgioFileNotFoundError,
    open_ome_zarr_container,
)
from bioio import BioImage
from bioio_ome_zarr import Reader as OmeZarrReader
from ngio.hcs._plate import OmeZarrPlate
import zarr
import numpy as np

from blimp.ome_ngff.layout import FieldLayout, _WELL_NAME_RE, _build_fov_roi_table
from blimp.ome_ngff.metadata import (
    NGFF_VERSION,
    _downsample_yx,
    _build_ngff_v05_metadata,
)

logger = logging.getLogger(__name__)

_PLATE_ROWS = {"96": list(string.ascii_uppercase[:8]), "384": list(string.ascii_uppercase[:16])}
_PLATE_COLUMNS = {"96": list(range(1, 13)), "384": list(range(1, 25))}


def ensure_plate_exists(
    plate_path: Union[str, Path], plate_name: str, plate_size: Literal["96", "384"] = "384"
) -> OmeZarrPlate:
    """Idempotently open or create a shared OME-Zarr plate store.

    Safe to call from multiple processes: if two callers race to create the
    same plate for the first time, the loser's ``create_empty_plate`` call
    fails and falls back to opening what the winner created.

    Pre-declares the full row/column grid (rows "A".."H"/"P", columns
    1-12/1-24 for a 96-/384-well plate) up front, independent of which wells
    actually get images. This costs nothing in storage (the plate store
    holds only its own small ``zarr.json`` until a well is actually
    written), and lets a viewer place populated wells at their true grid
    position rather than compacting the grid to only the wells a given run
    happened to touch.

    Intended to be called once, before any per-well writers run (e.g. from
    the same serial pass that discovers input files and generates per-well
    PBS jobscripts), rather than from every worker -- ``atomic_add_image``
    (used by the per-source writers) only guards concurrent *modification*
    of an existing plate, not its first creation.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    plate_name
        Name recorded in the plate's metadata (only used if the store does
        not already exist).
    plate_size
        "96" or "384" -- which standard plate row/column grid to declare
        (only used if the store does not already exist).

    Returns
    -------
    OmeZarrPlate
        Opened for writing (``mode="r+"``).
    """
    try:
        return open_ome_zarr_plate(store=str(plate_path), mode="r+")
    except NgioFileNotFoundError:
        pass

    try:
        plate = create_empty_plate(store=str(plate_path), name=plate_name, ngff_version=NGFF_VERSION, overwrite=False)
    except NgioFileExistsError:
        return open_ome_zarr_plate(store=str(plate_path), mode="r+")

    for row in _PLATE_ROWS[plate_size]:
        plate.add_row(row)
    for column in _PLATE_COLUMNS[plate_size]:
        plate.add_column(column)
    return plate


def locate_well(plate_path: Union[str, Path], well_name: str) -> str:
    """Resolve a well name to its path within a plate store.

    Prints (and returns) the well's path so it can be used directly in an
    ``rsync``/``scp`` command run from the *local* machine pulling from the
    server -- this function only resolves a path, it never itself shells
    out to copy anything, since the server generally has no outbound
    SSH/credentials to reach an arbitrary client.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    well_name
        e.g. ``"C09"``.

    Returns
    -------
    str
        The well's path relative to the plate store, e.g. ``"C/09"``.
    """
    well_match = _WELL_NAME_RE.match(well_name)
    if well_match is None:
        raise ValueError(f"Could not parse well name {well_name!r}, expected e.g. 'C09'")
    row, column = well_match.group(1).upper(), int(well_match.group(2))

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    well_path = plate.meta.get_well_path(row=row, column=column)
    full_path = str(Path(plate_path) / well_path)
    print(full_path)
    return full_path


def open_well_image(plate_path: Union[str, Path], well_relative_path: str, kind: Literal["stack", "mip"]) -> BioImage:
    """Open one well's "stack" or "mip" image (e.g. ``plate.zarr/C/09/mip``)
    as a ``BioImage``.

    ``BioImage(path)``'s automatic plugin resolution is extension-based (it
    only tries the ``bioio-ome-zarr`` reader for paths ending in ``.zarr``,
    ``.ozx``, or ``.zip``), so it never matches a well's path, which sits
    *inside* a plate store rather than being its own ``.zarr``-suffixed
    directory. Passing the reader explicitly bypasses that extension check.

    Parameters
    ----------
    plate_path
        Full path to the shared plate .zarr store.
    well_relative_path
        The well's path relative to the plate store, e.g. ``"C/09"`` (as
        returned by :func:`locate_well`).
    kind
        Which image to open -- "stack" (full z-stack) or "mip" (maximum
        intensity projection). Either, both, or neither may have been
        written for a given well.

    Returns
    -------
    BioImage

    Raises
    ------
    FileNotFoundError
        If ``kind`` was not written for this well.
    """
    well_path = Path(plate_path) / well_relative_path
    image_path = well_path / kind
    if not image_path.exists():
        well_group = zarr.open_group(str(well_path), mode="r")
        available = [image["path"] for image in well_group.attrs["ome"]["well"]["images"]]
        other_flag = "mip=True" if kind == "mip" else "keep_stacks=True"
        raise FileNotFoundError(
            f"Well {well_relative_path!r} has no {kind!r} image (has: {available!r}). "
            f"Re-run conversion with {other_flag} to add it."
        )
    return BioImage(str(image_path), reader=OmeZarrReader)


def _write_well_image(
    get_tile: Callable[[int], np.ndarray],
    layout: FieldLayout,
    plate: OmeZarrPlate,
    plate_path: Union[str, Path],
    image_path: str,
    channel_names: List[str],
    channel_colors: List[str],
    dtype: Any,
    num_levels: int,
    project_z: bool,
) -> OmeZarrContainer:
    """Register and write one image (full stack or MIP) for a well, then
    attach a "FOV_ROI_table" recording each original field's pixel region
    within the stitched canvas (see :func:`blimp.ome_ngff.layout._build_fov_roi_table`).

    Source-agnostic: ``get_tile(field_index)`` supplies each field's TCZYX
    pixel array (an nd2-sourced caller wraps ``BioImage.set_scene``; a
    TIFF-sourced caller opens each field's own TIFF, or returns a blank
    array for a field whose TIFF is missing).

    Parameters
    ----------
    get_tile
        Given a 0-indexed field index (in the same order as
        ``layout.offsets``), returns that field's TCZYX pixel array.
    layout
        As returned by ``get_field_layout``/``get_field_layout_from_tiff_metadata``.
    plate
        The already-open plate (see :func:`ensure_plate_exists`).
    plate_path
        Full path to the plate's .zarr store.
    image_path
        "stack" or "mip" (or any other short, alphanumeric path -- only
        ``ngio``'s ``path_in_well_validation`` constrains this).
    channel_names, channel_colors
        One entry per channel.
    dtype
        Pixel dtype for the written arrays.
    num_levels
        Number of pyramid levels to write.
    project_z
        Whether to write a maximum-intensity projection (True) or the full
        z-stack (False).

    Returns
    -------
    OmeZarrContainer
        The newly-written image, opened -- callers reuse it to attach
        labels/features without a redundant re-open.
    """
    canvas_shape = layout.canvas_shape
    if project_z:
        canvas_shape = (canvas_shape[0], canvas_shape[1], 1, canvas_shape[3], canvas_shape[4])

    logger.info(f"Registering well {layout.row}{layout.column:02d} image '{image_path}' in {plate_path}")
    well_relative_path = plate.atomic_add_image(row=layout.row, column=layout.column, image_path=image_path)

    image_name = f"{layout.row}{layout.column:02d}" + ("_mip" if project_z else "")
    attributes = _build_ngff_v05_metadata(
        image_name=image_name,
        num_levels=num_levels,
        pixel_size_x=layout.pixel_size_x,
        pixel_size_y=layout.pixel_size_y,
        pixel_size_z=layout.pixel_size_z,
        channel_names=channel_names,
        channel_colors=channel_colors,
    )
    attributes["blimp"] = {"image_kind": "mip" if project_z else "stack"}

    root_store = zarr.storage.LocalStore(str(plate_path))
    image_group = zarr.open_group(
        store=root_store,
        path=well_relative_path,
        mode="a",
        zarr_format=3,
        attributes=attributes,
    )

    level_shapes = []
    for level in range(num_levels):
        factor = 2**level
        # Ceiling division, matching what _downsample_yx's striding actually
        # produces (e.g. a 4691-pixel axis strided by 2 yields 2346 pixels,
        # not floor(4691/2) = 2345). zarr's array assignment silently clips a
        # too-large source to the destination shape rather than raising, so a
        # floor-divided level_shape here would quietly drop the last
        # row/column at every level instead of failing loudly.
        level_shapes.append(
            (
                canvas_shape[0],
                canvas_shape[1],
                canvas_shape[2],
                max(1, -(-canvas_shape[3] // factor)),
                max(1, -(-canvas_shape[4] // factor)),
            )
        )

    level_arrays = []
    for level, level_shape in enumerate(level_shapes):
        chunk_shape = (
            1,
            1,
            1,
            min(layout.tile_shape[3], level_shape[3]),
            min(layout.tile_shape[4], level_shape[4]),
        )
        level_arrays.append(
            image_group.create_array(
                name=str(level),
                shape=level_shape,
                dtype=dtype,
                chunks=chunk_shape,
                shards=None,
                dimension_names=["t", "c", "z", "y", "x"],
            )
        )

    for field_index, (y0, x0) in enumerate(layout.offsets):
        logger.debug(f"Writing field {field_index} at offset ({y0}, {x0})")
        tile = get_tile(field_index)
        if project_z:
            tile = np.max(tile, axis=2, keepdims=True)

        h, w = tile.shape[3], tile.shape[4]
        level_arrays[0][:, :, :, y0 : y0 + h, x0 : x0 + w] = tile

    logger.debug("Building pyramid levels")
    current = level_arrays[0][:, :, :, :, :]
    for level in range(1, num_levels):
        current = _downsample_yx(current)
        level_arrays[level][:, :, :, :, :] = current

    logger.debug(f"Attaching FOV_ROI_table to '{image_path}'")
    container = open_ome_zarr_container(str(Path(plate_path) / well_relative_path))
    container.add_table("FOV_ROI_table", _build_fov_roi_table(layout), overwrite=True)
    return container
