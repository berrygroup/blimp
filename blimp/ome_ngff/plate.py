"""Shared OME-Zarr plate/well registration and image writing, used by every
OME-NGFF writer (nd2-sourced, TIFF-sourced, and future Operetta-sourced)."""
from typing import Any, Dict, List, Union, Literal, Callable, Optional
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
import dask.array as da

from blimp.ome_ngff.labels import well_label_offset
from blimp.ome_ngff.layout import FieldLayout, _WELL_NAME_RE, _build_fov_roi_table
from blimp.ome_ngff.metadata import (
    NGFF_VERSION,
    _downsample_yx,
    _build_ngff_v05_metadata,
)

logger = logging.getLogger(__name__)

_PLATE_ROWS = {"96": list(string.ascii_uppercase[:8]), "384": list(string.ascii_uppercase[:16])}
_PLATE_COLUMNS = {"96": list(range(1, 13)), "384": list(range(1, 25))}


def resolve_plate_path(plate_path: Union[str, Path]) -> Path:
    """Resolve a user-given path to the actual plate .zarr store location.

    A path already ending in ``.zarr`` is used as-is; anything else is
    treated as a parent directory, inside which the store is named
    ``plate.zarr``. Pass the result to :func:`ensure_plate_exists` (using
    its own ``.stem`` as the plate name) so a bare folder still ends up
    with a sensible, derived name rather than the literal string "plate":
    ``-o some/experiment.zarr`` names the plate "experiment"; ``-o
    some/experiment`` places it at ``some/experiment/plate.zarr``, named
    "plate".
    """
    plate_path = Path(plate_path)
    return plate_path if plate_path.suffix == ".zarr" else plate_path / "plate.zarr"


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
        try:
            return open_ome_zarr_plate(store=str(plate_path), mode="r+")
        except NgioFileNotFoundError as e:
            # create_empty_plate refused because the path is non-empty, but
            # what's there isn't a valid OME-Zarr plate store either (e.g.
            # plate_path is an unrelated, already-populated directory) --
            # surface one clear error instead of this confusing pairing.
            raise FileExistsError(
                f"{plate_path} already exists and is not empty, but does not contain a valid "
                "OME-Zarr plate store. Pass a path to a new location, or to an existing plate "
                "store created by this same pipeline. If you're sure it's safe to remove and "
                f"want to rebuild fresh at this location, delete it yourself first: rm -rf {plate_path}"
            ) from e

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


def build_plate_pyramid(
    plate_path: Union[str, Path],
    kind: Literal["stack", "mip"] = "mip",
    label_name: Optional[str] = None,
    gap_fraction: float = 0.05,
) -> List[da.Array]:
    """Every pyramid level of the whole plate, as one lazy dask array per
    level, for viewing (or otherwise computing over) all wells at once at
    their true row/column position.

    Real wells are placed at their true grid position, each surrounded by a
    small empty margin (``gap_fraction`` of its own tile size, computed
    separately per pyramid level) so adjacent wells stay visually distinct;
    every other declared grid position is a zero-cost ``da.zeros``
    placeholder. ``da.zeros`` is defined analytically -- it never touches
    individual chunks at construction time -- and overlaying real wells via
    chunk-aligned ``__setitem__`` only swaps which task computes those
    chunks, so this costs ``O(populated wells)``, not ``O(declared grid
    size)``, regardless of how sparse the plate is. This is what
    ``ome_zarr.reader``'s own whole-plate stitching does not do -- it
    eagerly allocates real zero arrays for every declared-but-empty
    position, at every level and channel.

    Every well is read through ``ngio``'s own ``OmeZarrPlate``/
    ``OmeZarrContainer``/``Image`` API -- there's no separate "well zarr" to
    open by hand; a well is just a subgroup of the same plate store.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    kind
        Which image to build the pyramid from -- "stack" or "mip". Wells
        that don't have this image are skipped.
    label_name
        Build a label's pyramid instead of the intensity image's. Every
        well's own IDs get a well-specific offset first
        (:func:`blimp.ome_ngff.labels.well_label_offset`), so IDs are
        unique across the whole plate, not just within one well -- the
        canvas dtype becomes ``int64`` for this reason (wells without this
        label are skipped).
    gap_fraction
        Size of the empty margin between adjacent wells, as a fraction of
        each pyramid level's own tile size.

    Returns
    -------
    List[dask.array.Array]
        One array per pyramid level, finest first.

    Raises
    ------
    ValueError
        If no well has the requested image/label.
    """
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")

    containers: Dict[str, OmeZarrContainer] = {}
    for well_path in plate.wells_paths():
        row, column = well_path.split("/")
        try:
            container = plate.get_image(row, column, kind)
        except ValueError:
            continue
        if label_name is not None and label_name not in container.list_labels():
            continue
        containers[well_path] = container

    if not containers:
        what = f"label {label_name!r}" if label_name is not None else f"{kind!r} image"
        raise ValueError(f"No well in {plate_path} has a {what}.")

    def _get_array(container: OmeZarrContainer, level_path: str, offset: int = 0) -> da.Array:
        # Drop the leading T axis (always size 1 for blimp's images -- these
        # are static plates, not time series); keep C/Z (or just Z, for a
        # label) exactly as ngio returns them, so this works unchanged for
        # both a MIP (Z size 1) and a full stack (Z size N).
        if label_name is not None:
            arr = container.get_label(label_name, path=level_path).get_as_dask()[0]
            if offset:
                arr = arr.astype(np.int64)
                arr = da.where(arr == 0, 0, arr + offset)
            return arr
        return container.get_image(path=level_path).get_as_dask()[0]

    n_rows, n_cols = len(plate.rows), len(plate.columns)
    row_col_index = {}
    well_offsets = {}
    for well_path in containers:
        row, column = well_path.split("/")
        row_idx, col_idx = plate.rows.index(row), plate.columns.index(column)
        row_col_index[well_path] = (row_idx, col_idx)
        well_offsets[well_path] = well_label_offset(row_idx, col_idx) if label_name is not None else 0

    first_container = next(iter(containers.values()))
    pyramid = []
    for level_path in first_container.level_paths:
        reference = _get_array(first_container, level_path)
        *leading_shape, tile_h, tile_w = reference.shape
        pitch_h = round(tile_h * (1 + gap_fraction))
        pitch_w = round(tile_w * (1 + gap_fraction))

        canvas_dtype = np.int64 if label_name is not None else reference.dtype
        canvas = da.zeros(
            (*leading_shape, n_rows * pitch_h, n_cols * pitch_w),
            dtype=canvas_dtype,
            chunks=(*leading_shape, pitch_h, pitch_w),
        )
        for well_path, container in containers.items():
            row_idx, col_idx = row_col_index[well_path]
            y0, x0 = row_idx * pitch_h, col_idx * pitch_w
            canvas[..., y0 : y0 + tile_h, x0 : x0 + tile_w] = _get_array(container, level_path, well_offsets[well_path])
        pyramid.append(canvas)

    return pyramid


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
