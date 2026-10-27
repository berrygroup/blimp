"""Shared OME-Zarr plate/well registration and image writing, used by every
OME-NGFF writer (nd2-sourced, TIFF-sourced, and future Operetta-sourced)."""
from typing import Any, Dict, List, Union, Literal, Callable, Optional
from pathlib import Path
import time
import random
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
import pandas as pd
import filelock
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
    treated as a parent directory, with the store placed at
    ``<path>/plate.zarr``. Pass the result to :func:`ensure_plate_exists`
    (using its own ``.stem`` as the plate name) so a bare folder still gets
    a sensible derived name rather than the literal string "plate".

    Parameters
    ----------
    plate_path
        User-given path, e.g. from a CLI's ``-o``/``--plate_path``.

    Returns
    -------
    Path
        ``plate_path`` itself if it ends in ``.zarr``, else
        ``plate_path/plate.zarr``.
    """
    plate_path = Path(plate_path)
    return plate_path if plate_path.suffix == ".zarr" else plate_path / "plate.zarr"


def ensure_plate_exists(
    plate_path: Union[str, Path], plate_name: str, plate_size: Literal["96", "384"] = "384"
) -> OmeZarrPlate:
    """Idempotently open or create a shared OME-Zarr plate store.

    Safe to call from multiple processes: if two callers race to create the
    same plate, the loser's ``create_empty_plate`` call fails and falls
    back to opening what the winner created. Pre-declares the full
    row/column grid up front (costs nothing in storage until a well is
    actually written), so a viewer can place populated wells at their true
    grid position rather than only the wells a given run happened to
    touch. Call once, before any per-well writers run --
    ``atomic_add_image`` (used by the per-source writers) only guards
    concurrent *modification* of an existing plate, not its first creation.

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


def _discover_wells_with_label(
    plate: OmeZarrPlate,
    kind: Literal["stack", "mip"],
    label_name: Optional[str] = None,
    wells: Optional[Union[str, List[str]]] = None,
) -> Dict[str, OmeZarrContainer]:
    """Open every well's own ``kind`` image, restricted to wells that have
    ``label_name`` (if given) and, optionally, to an explicit ``wells``
    subset -- shared by ``build_plate_pyramid``, ``build_feature_pyramid``,
    and ``_read_plate_wide_features``.

    Parameters
    ----------
    plate
        The already-open plate.
    kind
        "stack" or "mip". Wells that don't have this image are skipped.
    label_name
        Only keep wells that have this label. ``None`` keeps every well
        that has the ``kind`` image, regardless of labels.
    wells
        Restrict discovery to one well path (e.g. ``"C/09"``, the same
        format ``plate.wells_paths()`` returns) or a list of them. ``None``
        (the default) considers every well the plate has.

    Returns
    -------
    Dict[str, OmeZarrContainer]
        Keyed by well path (e.g. ``"C/09"``).

    Raises
    ------
    ValueError
        If ``wells`` names a well path not in ``plate.wells_paths()``.
    """
    all_well_paths = plate.wells_paths()
    if wells is None:
        well_paths = all_well_paths
    else:
        requested = [wells] if isinstance(wells, str) else list(wells)
        unknown = sorted(set(requested) - set(all_well_paths))
        if unknown:
            raise ValueError(f"Well(s) {unknown} not found in plate; available wells: {sorted(all_well_paths)}")
        well_paths = requested

    containers: Dict[str, OmeZarrContainer] = {}
    for well_path in well_paths:
        row, column = well_path.split("/")
        try:
            container = plate.get_image(row, column, kind)
        except ValueError:
            continue
        if label_name is not None and label_name not in container.list_labels():
            continue
        containers[well_path] = container
    return containers


def build_plate_pyramid(
    plate_path: Union[str, Path],
    kind: Literal["stack", "mip"] = "mip",
    label_name: Optional[str] = None,
    gap_fraction: float = 0.05,
) -> List[da.Array]:
    """Every pyramid level of the whole plate, as one lazy dask array per
    level, for viewing (or otherwise computing over) all wells at once at
    their true row/column position.

    Real wells are placed at their true grid position, with a small empty
    margin (``gap_fraction`` of tile size) between them; every other
    declared grid position is a zero-cost ``da.zeros`` placeholder, so cost
    scales with ``O(populated wells)``, not the declared grid size --
    unlike ``ome_zarr.reader``'s own whole-plate stitching, which eagerly
    allocates a real zero array for every declared-but-empty position, at
    every level and channel. Every well is read through ``ngio``'s own
    ``OmeZarrPlate``/``OmeZarrContainer``/``Image`` API.

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
    containers = _discover_wells_with_label(plate, kind, label_name)

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


def _read_plate_wide_features(
    plate_path: Union[str, Path],
    label_name: str,
    kind: Literal["stack", "mip"] = "mip",
    wells: Optional[Union[str, List[str]]] = None,
) -> Optional[pd.DataFrame]:
    """Every contributing well's own ``f"{label_name}_features"`` table,
    merged into one plate-wide dataframe with its ``"label"`` column
    rewritten to match that well's own plate-wide-unique pixel values
    (see :func:`blimp.ome_ngff.labels.well_label_offset`) -- prefers each
    well's already-persisted ``global_id_numeric`` column when present,
    else derives the identical value on the fly.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to read (``f"{label_name}_features"``).
    kind
        "stack" or "mip".
    wells
        Restrict to one well path (e.g. ``"C/09"``) or a list of them (see
        :func:`_discover_wells_with_label`). ``None`` (the default) merges
        every well that has both the label and a matching features table.

    Returns
    -------
    Optional[pandas.DataFrame]
        ``None`` if no (selected) well has a matching features table.
    """
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, wells)

    feature_frames = []
    for well_path, container in containers.items():
        table_name = f"{label_name}_features"
        if table_name not in container.list_tables():
            continue
        row, column = well_path.split("/")
        df = container.get_feature_table(table_name).dataframe.reset_index()
        if "global_id_numeric" in df.columns:
            df["label"] = df["global_id_numeric"]
        else:
            df["label"] = df["label"] + well_label_offset(plate.rows.index(row), plate.columns.index(column))
        feature_frames.append(df)

    if not feature_frames:
        return None
    return pd.concat(feature_frames, ignore_index=True)


def build_feature_pyramid(
    plate_path: Union[str, Path],
    label_name: str,
    feature_name: str,
    kind: Literal["stack", "mip"] = "mip",
    gap_fraction: float = 0.05,
    wells: Optional[Union[str, List[str]]] = None,
) -> List[da.Array]:
    """Every pyramid level of a plate-wide *feature-value* image: each
    object's own pixels hold its own ``feature_name`` measurement (a plain
    float) instead of a label ID -- independent of per-label colormap
    tools like Napari Feature Visualizer, which assume label IDs are
    packed near zero and can run out of memory on plate-wide-unique IDs
    (see :func:`blimp.ome_ngff.labels.well_label_offset`).

    Background pixels, and any object with no matching feature row, read
    as ``NaN`` (the canvas dtype is always ``float32``). Cost scales with
    the number of *populated* (and, if given, *selected*) wells, not the
    plate's declared grid size, same as ``build_plate_pyramid`` -- pass
    ``wells`` to restrict this to only the well(s) you actually need.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to show (``f"{label_name}_features"``).
    feature_name
        Which column of that features table to show.
    kind
        "stack" or "mip".
    gap_fraction
        Size of the empty margin between adjacent wells, as a fraction of
        each pyramid level's own tile size (see ``build_plate_pyramid``).
    wells
        Restrict to one well path (e.g. ``"C/09"``) or a list of them (see
        :func:`_discover_wells_with_label`). ``None`` (the default)
        includes every populated well.

    Returns
    -------
    List[dask.array.Array]
        One array per pyramid level, finest first.

    Raises
    ------
    ValueError
        If no (selected) well has a ``f"{label_name}_features"`` table, or
        if ``feature_name`` isn't one of its columns.
    """
    features_df = _read_plate_wide_features(plate_path, label_name, kind, wells)
    if features_df is None:
        raise ValueError(f"No well in {plate_path} has a {label_name}_features table.")
    if feature_name not in features_df.columns:
        raise ValueError(
            f"{feature_name!r} is not a column of {label_name}_features; available: {sorted(features_df.columns)}"
        )

    order = np.argsort(features_df["label"].to_numpy())
    sorted_ids = features_df["label"].to_numpy()[order]
    sorted_values = features_df[feature_name].to_numpy(dtype=np.float32)[order]

    def _remap(block: np.ndarray) -> np.ndarray:
        flat = block.ravel()
        idx = np.clip(np.searchsorted(sorted_ids, flat), 0, len(sorted_ids) - 1)
        matched = sorted_ids[idx] == flat
        return np.where(matched, sorted_values[idx], np.nan).astype(np.float32).reshape(block.shape)

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, wells)

    n_rows, n_cols = len(plate.rows), len(plate.columns)
    row_col_index = {}
    well_offsets = {}
    for well_path in containers:
        row, column = well_path.split("/")
        row_idx, col_idx = plate.rows.index(row), plate.columns.index(column)
        row_col_index[well_path] = (row_idx, col_idx)
        well_offsets[well_path] = well_label_offset(row_idx, col_idx)

    first_container = next(iter(containers.values()))
    pyramid = []
    for level_path in first_container.level_paths:
        reference = first_container.get_label(label_name, path=level_path).get_as_dask()[0]
        *leading_shape, tile_h, tile_w = reference.shape
        pitch_h = round(tile_h * (1 + gap_fraction))
        pitch_w = round(tile_w * (1 + gap_fraction))

        canvas = da.full(
            (*leading_shape, n_rows * pitch_h, n_cols * pitch_w),
            np.nan,
            dtype=np.float32,
            chunks=(*leading_shape, pitch_h, pitch_w),
        )
        for well_path, container in containers.items():
            row_idx, col_idx = row_col_index[well_path]
            y0, x0 = row_idx * pitch_h, col_idx * pitch_w
            arr = container.get_label(label_name, path=level_path).get_as_dask()[0]
            arr = da.where(arr == 0, 0, arr.astype(np.int64) + well_offsets[well_path])
            canvas[..., y0 : y0 + tile_h, x0 : x0 + tile_w] = arr.map_blocks(_remap, dtype=np.float32)
        pyramid.append(canvas)

    return pyramid


def _atomic_add_image_with_retry(
    plate: OmeZarrPlate,
    row: str,
    column: int,
    image_path: str,
    max_attempts: int = 5,
    initial_backoff: float = 1.0,
) -> str:
    """Call ``plate.atomic_add_image``, retrying with exponential backoff
    (plus jitter) if it fails because the plate's own lock couldn't be
    acquired.

    ``atomic_add_image``'s internal lock has a fixed 10-second timeout
    inside ``ngio`` itself (``ngio/utils/_zarr_utils.py``'s
    ``FileLock(_lock_path, timeout=10)`` -- no parameter blimp or any other
    caller can override). A PBS array job registers many wells into the
    same plate concurrently, one per task; under heavy enough contention
    for this single shared lock, a task can exhaust that fixed timeout and
    crash outright rather than simply waiting its turn. Retrying here (ngio
    exposes no retry hook for this lock) gives contention time to clear
    across the whole array job, at the cost of a longer worst-case runtime
    for whichever tasks actually hit it.

    Parameters
    ----------
    plate
        The already-open plate.
    row, column, image_path
        Passed straight through to ``plate.atomic_add_image``.
    max_attempts
        Total attempts before giving up and letting the ``filelock.Timeout``
        propagate.
    initial_backoff
        Seconds to wait before the second attempt; doubles (plus up to 50%
        random jitter, to desynchronize many simultaneously-retrying tasks
        rather than have them all wake up and collide again at once) each
        attempt after that.

    Returns
    -------
    str
        The well's path, as returned by ``atomic_add_image``.
    """
    for attempt in range(1, max_attempts):
        try:
            return plate.atomic_add_image(row=row, column=column, image_path=image_path)
        except filelock.Timeout:
            backoff = initial_backoff * (2 ** (attempt - 1)) * (1 + random.random() * 0.5)
            logger.warning(
                f"Timed out acquiring the plate's own lock while registering well "
                f"{row}{column:02d} (attempt {attempt}/{max_attempts}); retrying in "
                f"{backoff:.1f}s. Expected under heavy concurrent write load (e.g. many "
                "PBS array tasks writing to the same plate at once) -- ngio's own lock "
                "timeout is a fixed 10s, not configurable."
            )
            time.sleep(backoff)
    return plate.atomic_add_image(row=row, column=column, image_path=image_path)


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
    well_relative_path = _atomic_add_image_with_retry(plate, layout.row, layout.column, image_path)

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
