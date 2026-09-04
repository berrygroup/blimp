"""Assemble a whole-plate OME-NGFF (OME-Zarr) store from an existing
OME-TIFF pipeline: per-field intensity TIFFs (from ``nd2_to_ome_tiff.py``),
per-field segmentation label TIFFs, and per-field ``quantify()`` measurement
CSVs.

Input contract (matches the lab's own established convention -- see
``run_segment_and_quantify.py`` in the ``berrygroup/publications`` repo):
one intensity TIFF, one (possibly multi-channel) label TIFF, and one
already-aggregated measurements CSV per field, all sharing the intensity
TIFF's own filename stem, each in their own directory (e.g.
``OME-TIFF-MIP/``, ``SEGMENTATION/``, ``QUANTIFICATION/``). A label TIFF's
own channel names (e.g. ``["Nuclei", "Cell"]``) name the objects it
segments -- every channel is written as its own label layer, but only the
one channel `quantify()`'s own aggregation was built around (its
``parent_label_name``, see :func:`_get_parent_channel_name`) gets a
``FeatureTable``, since a single aggregated CSV already folds every child
object's stats into that one parent's own rows.

Real stage positions are read from the metadata sidecar ``nd2_to_ome_tiff.py``
writes alongside the field TIFFs (only when called with ``mip=True`` or
``keep_stacks=True``) -- see ``nd2_parse_metadata.py::nd2_extract_metadata_and_save``.
Placement is "grid" (default) or "exact", same choice and same meaning
as ``nd2_to_ome_ngff.py`` -- see :func:`get_field_layout_from_tiff_metadata`.

Robust to partial upstream failure, field by field: a missing intensity or
label TIFF is substituted with a blank (all-zero) array of that field's tile
shape; a missing measurements CSV simply contributes no rows to the
parent's feature table for that field. The metadata sidecar's own field
list is authoritative for "which fields should exist," independent of
which downstream files actually landed on disk -- see
:func:`_discover_well_manifest`.
"""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging

from ngio import open_ome_zarr_plate
from bioio import BioImage
import numpy as np
import pandas as pd

from blimp.log import configure_logging
from blimp.ome_ngff import NUM_PYRAMID_LEVELS, ensure_plate_exists
from blimp.ome_ngff.plate import _write_well_image
from blimp.ome_ngff.labels import _write_well_labels, _write_well_points
from blimp.ome_ngff.layout import (
    FieldLayout,
    _parse_well_name,
    _cluster_grid_index,
    _exact_pixel_offset,
)
from blimp.ome_ngff.features import _write_well_features

logger = logging.getLogger(__name__)

# A source TIFF pipeline never carries channel-color metadata (confirmed:
# nd2_to_ome_tiff.py's OmeTiffWriter.save() call never passes one), so the
# TIFF-sourced layout always falls back to this cycle -- not a defensive
# "just in case" branch, an expected characteristic of this input.
_DEFAULT_CHANNEL_COLORS = ["FF0000", "00FF00", "0000FF", "FF00FF", "00FFFF", "FFFF00"]


def _metadata_csv_path(nd2_stem: str, tiff_dir: Union[str, Path]) -> Path:
    return Path(tiff_dir) / f"{nd2_stem}_metadata.csv"


def _read_metadata_csv(nd2_stem: str, tiff_dir: Union[str, Path]) -> pd.DataFrame:
    """Read and dedupe the ``nd2_to_ome_tiff.py`` metadata sidecar to one
    row per ``field_id``, sorted ascending.

    Raises
    ------
    FileNotFoundError
        If the sidecar doesn't exist -- naming the flag that produces it,
        since it's easy to have converted without it (``mip=False`` and
        ``keep_stacks=False``).
    """
    metadata_csv_path = _metadata_csv_path(nd2_stem, tiff_dir)
    if not metadata_csv_path.exists():
        raise FileNotFoundError(
            f"No metadata sidecar found at {metadata_csv_path}. This is written by "
            "nd2_to_ome_tiff(...) only when called with mip=True or keep_stacks=True -- "
            "re-run the TIFF conversion with one of those flags to produce it."
        )
    df = pd.read_csv(metadata_csv_path)
    return df.drop_duplicates(subset="field_id").sort_values("field_id").reset_index(drop=True)


def get_field_layout_from_tiff_metadata(
    nd2_stem: str,
    tiff_dir: Union[str, Path],
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
) -> FieldLayout:
    """Compute the stitching layout for a well's fields of view from
    ``nd2_to_ome_tiff.py``'s own metadata sidecar, rather than from an nd2
    file directly.

    The raw ``stage_x_abs``/``stage_y_abs`` columns feed the same layout
    math ``nd2_to_ome_ngff.py`` uses -- not ``standard_field_id`` (a single
    raster-order integer, not a row/column pair, so it can't drive tile
    placement directly).

    Parameters
    ----------
    nd2_stem
        The source nd2 file's stem, e.g. ``"WellC09_Channel647..."`` --
        shared by the metadata sidecar (``{nd2_stem}_metadata.csv``) and
        every field TIFF's own filename.
    tiff_dir
        Directory containing the field TIFFs and the metadata sidecar (e.g.
        an ``OME-TIFF-MIP/`` folder).
    y_direction, x_direction
        See :func:`blimp.preprocessing.nd2_to_ome_ngff.get_field_layout` --
        same convention, same defaults.
    placement
        "grid" or "exact" -- see
        :func:`blimp.preprocessing.nd2_to_ome_ngff.get_field_layout`, same
        meaning.

    Returns
    -------
    FieldLayout

    Raises
    ------
    FileNotFoundError
        If the metadata sidecar, or every field TIFF it lists, is missing.
    """
    if y_direction not in {"up", "down"}:
        raise ValueError(f'y_direction = {y_direction}, only "up" or "down" are possible')
    if x_direction not in {"left", "right"}:
        raise ValueError(f'x_direction = {x_direction}, only "left" or "right" are possible')
    if placement not in {"grid", "exact"}:
        raise ValueError(f'placement = {placement}, only "grid" or "exact" are possible')

    df = _read_metadata_csv(nd2_stem, tiff_dir)
    field_ids = df["field_id"].tolist()
    stage_x = df["stage_x_abs"].to_numpy()
    stage_y = df["stage_y_abs"].to_numpy()

    first_tiff_path = None
    for filename in df["filename_ome_tiff"]:
        candidate = Path(tiff_dir) / filename
        if candidate.exists():
            first_tiff_path = candidate
            break
    if first_tiff_path is None:
        raise FileNotFoundError(
            f"None of the field TIFFs listed in {_metadata_csv_path(nd2_stem, tiff_dir)} "
            f"exist in {tiff_dir} -- cannot determine tile shape/dtype/pixel size."
        )

    reference_image = BioImage(str(first_tiff_path))
    tile_shape = reference_image.shape
    pixel_size = reference_image.physical_pixel_sizes

    if placement == "grid":
        tile_extent_x = tile_shape[4] * pixel_size.X
        tile_extent_y = tile_shape[3] * pixel_size.Y
        col_idx = _cluster_grid_index(stage_x, tile_extent_x)
        row_idx = _cluster_grid_index(stage_y, tile_extent_y)
        if x_direction == "left":
            col_idx = col_idx.max() - col_idx
        if y_direction == "up":
            row_idx = row_idx.max() - row_idx
        offset_x_px = col_idx * tile_shape[4]
        offset_y_px = row_idx * tile_shape[3]
    else:
        offset_x_px = _exact_pixel_offset(stage_x, pixel_size.X, reverse=(x_direction == "left"))
        offset_y_px = _exact_pixel_offset(stage_y, pixel_size.Y, reverse=(y_direction == "up"))
    offsets = list(zip(offset_y_px.tolist(), offset_x_px.tolist()))

    row, column = _parse_well_name(nd2_stem, [None])

    canvas_y = max(y0 for y0, _ in offsets) + tile_shape[3]
    canvas_x = max(x0 for _, x0 in offsets) + tile_shape[4]
    canvas_shape = (tile_shape[0], tile_shape[1], tile_shape[2], canvas_y, canvas_x)

    channel_names = list(reference_image.channel_names)
    logger.info("TIFF-sourced channels carry no color metadata; using a default color cycle.")
    channel_colors = [_DEFAULT_CHANNEL_COLORS[i % len(_DEFAULT_CHANNEL_COLORS)] for i in range(len(channel_names))]

    # No XYPosLoop-style position-name metadata exists for a TIFF source, so
    # synthesize one in the same "{well}_{field_id:04d}" shape nd2 uses
    # (e.g. "C09_0001"), giving FOV_ROI_table traceable names.
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


def _discover_well_manifest(
    nd2_stem: str,
    tiff_dir: Union[str, Path],
    label_dir: Optional[Union[str, Path]] = None,
    feature_csv_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Cross-reference the metadata sidecar's field list (authoritative --
    fixed at acquisition time, independent of which downstream files
    actually exist) against what's actually present in the intensity TIFF
    directory, the label directory, and the feature-CSV directory. Logs a
    per-source summary naming exactly which field IDs are missing, so gaps
    are observable rather than silently papered over.

    ``label_dir``/``feature_csv_dir`` hold one (possibly multi-channel)
    label TIFF / one (already-aggregated) feature CSV per *field* -- not one
    per named object -- so existence is a single column each, not one per
    channel name.

    Returns
    -------
    pd.DataFrame
        One row per ``field_id`` (plus ``stage_x_abs``/``stage_y_abs``/
        ``filename_ome_tiff`` from the sidecar), with an added
        ``intensity_exists`` boolean column, and (if given) ``label_exists``/
        ``feature_exists`` boolean columns.
    """
    df = _read_metadata_csv(nd2_stem, tiff_dir)
    stems = df["filename_ome_tiff"].apply(lambda f: Path(f).stem)

    df["intensity_exists"] = df["filename_ome_tiff"].apply(lambda f: (Path(tiff_dir) / f).exists())
    missing = df.loc[~df["intensity_exists"], "field_id"].tolist()
    if missing:
        logger.warning(f"Well {nd2_stem}: intensity TIFF missing for field_id(s) {missing}")

    if label_dir is not None:
        df["label_exists"] = df["filename_ome_tiff"].apply(lambda f: (Path(label_dir) / f).exists())
        missing = df.loc[~df["label_exists"], "field_id"].tolist()
        if missing:
            logger.warning(f"Well {nd2_stem}: label TIFF missing for field_id(s) {missing}")

    if feature_csv_dir is not None:
        df["feature_exists"] = stems.apply(lambda s: (Path(feature_csv_dir) / f"{s}.csv").exists())
        missing = df.loc[~df["feature_exists"], "field_id"].tolist()
        if missing:
            logger.warning(f"Well {nd2_stem}: feature CSV missing for field_id(s) {missing}")

    logger.info(
        f"Well {nd2_stem}: {len(df)} fields expected; "
        f"intensity found for {int(df['intensity_exists'].sum())}/{len(df)}"
    )
    return df


def _get_parent_channel_name(feature_csv_columns_df: pd.DataFrame) -> str:
    """The parent channel's own name, read directly off the
    ``parent_label_name`` column ``quantify()`` leaves on an aggregated
    result.

    ``quantify()``'s own top-level loop measures the parent object against
    *itself* as its own designated parent too (alongside every real child),
    and masking a channel's own label array with itself is trivially "every
    object is 100% contained within itself" -- so this column already names
    the parent channel directly, no inference needed.

    Raises
    ------
    ValueError
        If the column is absent (``quantify()`` was called without
        ``parent_object``) -- there is then no way to know which channel
        this CSV's measurements are about.
    """
    if "parent_label_name" not in feature_csv_columns_df.columns:
        raise ValueError(
            "No 'parent_label_name' column -- was quantify() called with parent_object? "
            "Without it there's no way to tell which label channel this feature CSV is about."
        )
    return feature_csv_columns_df["parent_label_name"].iloc[0]


def _is_point_object_channel(
    channel_name: str,
    parent_channel_name: Optional[str],
    feature_csv_columns_df: Optional[pd.DataFrame],
    point_object_channel_names: Optional[List[str]],
) -> bool:
    """Determine whether a label channel has no stable per-pixel identity
    and should become a ``GenericRoiTable`` (via ``_write_well_points``)
    rather than an ``ngio.Label`` (via ``_write_well_labels``).

    An explicit name wins first -- mirrors ``quantify()``'s own
    ``point_objects`` parameter, which the caller already had to specify
    explicitly when they ran ``quantify()`` in the first place. Otherwise,
    if a feature CSV is available: the parent channel's own status is its
    plain ``is_point_object`` column; any other channel's is its own
    ``f"{name}_is_point_object"`` column (see
    ``aggregate_and_merge_features``/``_quantify_point_object_aggregated_to_parent``
    in ``quantify.py``). Falls back to ``False`` (blob) with a logged
    warning if none of that is available -- point/blob-ness genuinely can't
    be determined from the label file alone.
    """
    if point_object_channel_names and channel_name in point_object_channel_names:
        return True
    if feature_csv_columns_df is not None:
        if channel_name == parent_channel_name and "is_point_object" in feature_csv_columns_df.columns:
            return bool(feature_csv_columns_df["is_point_object"].iloc[0])
        point_col = f"{channel_name}_is_point_object"
        if point_col in feature_csv_columns_df.columns:
            return bool(feature_csv_columns_df[point_col].iloc[0])
    logger.warning(
        f"Could not determine whether '{channel_name}' is a point-object channel "
        "(not named in point_object_channel_names, and no feature CSV column available); "
        "treating it as a regular (blob) object."
    )
    return False


def convert_tiff_well_to_ome_ngff(
    nd2_stem: str,
    tiff_dir: Union[str, Path],
    plate_path: Union[str, Path],
    label_dir: Optional[Union[str, Path]] = None,
    feature_csv_dir: Optional[Union[str, Path]] = None,
    point_object_channel_names: Optional[List[str]] = None,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    num_levels: int = NUM_PYRAMID_LEVELS,
) -> None:
    """Stitch one well's TIFF pipeline output (intensity, labels, features)
    into a well image in a shared OME-NGFF plate store.

    Parameters
    ----------
    nd2_stem
        The source nd2 file's stem shared by the metadata sidecar and every
        field TIFF's own filename (see :func:`get_field_layout_from_tiff_metadata`).
    tiff_dir
        Directory containing the intensity field TIFFs and the metadata
        sidecar.
    plate_path
        Full path to the shared plate .zarr store. Must already exist --
        see :func:`blimp.ome_ngff.ensure_plate_exists`.
    label_dir
        Directory containing one (possibly multi-channel) label TIFF per
        field (same filenames as the intensity TIFFs) -- each channel's own
        OME channel-name (e.g. ``"Nuclei"``, ``"Cell"``) names the object it
        segments, and every channel is written as its own label layer.
        ``None`` skips labels entirely. Each label TIFF's own Z depth must
        match ``tiff_dir``'s (both 1 for a MIP, or both equal for a real
        stack) -- ``quantify()`` and the rest of ``segment.py`` support 3D
        label images even though ``segment_nuclei_cellpose`` itself only
        produces 2D ones.
    feature_csv_dir
        Directory containing one already-aggregated ``quantify()``
        measurement CSV per field (same filename stems) -- there is exactly
        one CSV per field regardless of how many label channels exist,
        since aggregation folds every child object's stats into the
        parent's own rows. Which label channel this CSV's measurements
        belong to is read directly off its own ``parent_label_name`` column
        (see :func:`_get_parent_channel_name`) -- only that one channel gets
        a ``FeatureTable`` attached; every other channel is still written as
        a label layer with no measurements of its own.
    point_object_channel_names
        Names of label channels, if any, that have no stable per-pixel
        identity and should become a ``GenericRoiTable`` instead of an
        ``ngio.Label`` -- mirrors ``quantify()``'s own ``point_objects``
        parameter. Optional: when a ``feature_csv_dir`` is given, each
        channel's point/blob status is normally read directly off that CSV
        instead (see :func:`_is_point_object_channel`) -- this is a
        fallback/override for when it isn't (e.g. no ``feature_csv_dir`` at
        all), or to force a channel either way.
    y_direction, x_direction, placement
        See :func:`get_field_layout_from_tiff_metadata`.
    channel_names
        List of channel names in case those found in the TIFF metadata are
        incorrect.
    num_levels
        Number of pyramid levels to write.

    Notes
    -----
    Whether the written image is registered as ``"stack"`` or ``"mip"`` (and
    its ``blimp.image_kind`` attribute -- see
    :func:`blimp.ome_ngff.plate._write_well_image`) is auto-detected from
    ``tiff_dir``'s own field TIFFs, not chosen by the caller: single-Z
    fields (as in a ``*-MIP`` directory) write to ``"mip"``, multi-Z fields
    write to ``"stack"``. To end up with both a ``"stack"`` and a ``"mip"``
    image in the same well, call this function twice, once per source
    directory (mirroring how ``nd2_to_ome_ngff.py`` writes each
    independently under its own ``keep_stacks``/``mip`` flag).
    """
    logger.info(f"Reading layout for well {nd2_stem}")
    layout = get_field_layout_from_tiff_metadata(
        nd2_stem, tiff_dir, y_direction=y_direction, x_direction=x_direction, placement=placement
    )
    is_mip = layout.tile_shape[2] == 1
    manifest = _discover_well_manifest(nd2_stem, tiff_dir, label_dir, feature_csv_dir)

    if channel_names is None:
        channel_names = layout.channel_names
    elif isinstance(channel_names, str):
        channel_names = [channel_names]

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r+")

    manifest_by_field = manifest.set_index("field_id")

    def get_tile(field_index: int) -> np.ndarray:
        field_id = layout.field_ids[field_index]
        row = manifest_by_field.loc[field_id]
        if not row["intensity_exists"]:
            return np.zeros(layout.tile_shape, dtype=reference_dtype)
        path = Path(tiff_dir) / row["filename_ome_tiff"]
        return BioImage(str(path)).get_image_data("TCZYX")

    # dtype for blank-substitution: read from the first available field.
    first_available = manifest.loc[manifest["intensity_exists"], "filename_ome_tiff"]
    if first_available.empty:
        raise FileNotFoundError(f"No intensity TIFFs found at all for well {nd2_stem} in {tiff_dir}")
    reference_dtype = BioImage(str(Path(tiff_dir) / first_available.iloc[0])).dtype

    container = _write_well_image(
        get_tile=get_tile,
        layout=layout,
        plate=plate,
        plate_path=plate_path,
        image_path="mip" if is_mip else "stack",
        channel_names=channel_names,
        channel_colors=layout.channel_colors,
        dtype=reference_dtype,
        num_levels=num_levels,
        # A no-op when is_mip (already Z==1); tied to the same check used
        # for image_path so the two can never disagree.
        project_z=is_mip,
    )

    if label_dir is None:
        return

    # Channel names are read once, from the first available field's own
    # label TIFF -- the same "first available field" pattern
    # get_field_layout_from_tiff_metadata already uses for tile shape/pixel
    # size, on the assumption that every field's label TIFF shares the same
    # channel layout (one segmentation pipeline run).
    label_channel_names: Optional[List[str]] = None
    for field_id in layout.field_ids:
        row = manifest_by_field.loc[field_id]
        if row["label_exists"]:
            label_channel_names = list(BioImage(str(Path(label_dir) / row["filename_ome_tiff"])).channel_names)
            break
    if label_channel_names is None:
        raise FileNotFoundError(f"No label TIFFs found at all for well {nd2_stem} in {label_dir}")

    unknown_point_object_names = set(point_object_channel_names or []) - set(label_channel_names)
    if unknown_point_object_names:
        raise ValueError(
            f"point_object_channel_names {sorted(unknown_point_object_names)} not found among the label "
            f"TIFF's own channels {label_channel_names} for well {nd2_stem}"
        )

    # The first available feature CSV's own columns tell us which channel is
    # the aggregation parent (_get_parent_channel_name), and per-channel
    # point/blob status (_is_point_object_channel) -- read once and reused
    # for every channel below, since there's exactly one CSV per field.
    feature_csv_columns_df: Optional[pd.DataFrame] = None
    parent_channel_name: Optional[str] = None
    if feature_csv_dir is not None:
        for field_id in layout.field_ids:
            row = manifest_by_field.loc[field_id]
            if row["feature_exists"]:
                stem = Path(row["filename_ome_tiff"]).stem
                feature_csv_columns_df = pd.read_csv(Path(feature_csv_dir) / f"{stem}.csv")
                break
        if feature_csv_columns_df is None:
            raise FileNotFoundError(f"No feature CSVs found at all for well {nd2_stem} in {feature_csv_dir}")
        parent_channel_name = _get_parent_channel_name(feature_csv_columns_df)
        if parent_channel_name not in label_channel_names:
            logger.warning(
                f"Well {nd2_stem}: feature CSV's parent channel '{parent_channel_name}' is not among the "
                f"label TIFF's own channels {label_channel_names} -- no channel will get a FeatureTable."
            )

    well_name = f"{layout.row}{layout.column:02d}"

    for channel_index, label_name in enumerate(label_channel_names):
        is_point_object = _is_point_object_channel(
            label_name, parent_channel_name, feature_csv_columns_df, point_object_channel_names
        )
        is_parent = feature_csv_dir is not None and label_name == parent_channel_name

        # Point objects are read as flat 2D masks (_write_well_points has no
        # 3D support); blob objects keep their real Z depth so a genuine 3D
        # label places one slice per Z-plane rather than broadcasting a 2D
        # mask across every plane.
        field_arrays: Dict[int, Optional[np.ndarray]] = {}
        for field_id in layout.field_ids:
            row = manifest_by_field.loc[field_id]
            if not row["label_exists"]:
                field_arrays[field_id] = None
                continue
            path = Path(label_dir) / row["filename_ome_tiff"]
            label_image = BioImage(str(path))
            if is_point_object:
                field_arrays[field_id] = np.squeeze(label_image.get_image_data("YX", C=channel_index, T=0))
            else:
                field_arrays[field_id] = label_image.get_image_data("ZYX", C=channel_index, T=0)

        # Only the parent channel gets measurements attached -- every child
        # object's stats are already folded into the parent's own rows.
        field_features: Dict[int, Optional[pd.DataFrame]] = {field_id: None for field_id in layout.field_ids}
        if is_parent:
            for field_id in layout.field_ids:
                row = manifest_by_field.loc[field_id]
                if not row["feature_exists"]:
                    continue
                stem = Path(row["filename_ome_tiff"]).stem
                field_features[field_id] = pd.read_csv(Path(feature_csv_dir) / f"{stem}.csv")  # type: ignore[arg-type]

        if is_point_object:
            _write_well_points(
                container=container,
                layout=layout,
                well_name=well_name,
                channel_name=label_name,
                field_masks=field_arrays,
                field_features=field_features,
            )
        else:
            _write_well_labels(container=container, layout=layout, label_name=label_name, field_arrays=field_arrays)
            if is_parent:
                _write_well_features(
                    container=container, label_name=label_name, field_dataframes=field_features, well_name=well_name
                )


def tiff_to_ome_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    plate_name: Optional[str] = None,
    n_batches: int = 1,
    batch_id: int = 0,
    label_dir: Optional[Union[str, Path]] = None,
    feature_csv_dir: Optional[Union[str, Path]] = None,
    point_object_channel_names: Optional[List[str]] = None,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
) -> None:
    """Read a folder of field TIFFs + metadata sidecars (one well each,
    named ``{nd2_stem}_metadata.csv``) and stitch them into a shared
    OME-NGFF plate store, including any configured labels/features. Can
    perform batch processing (one well per batch unit).

    Note: this does not itself call :func:`blimp.ome_ngff.ensure_plate_exists`
    for every well -- it assumes the plate store already exists (call it
    once beforehand) so that concurrent batches never race to create it.

    Parameters
    ----------
    in_path
        Directory containing the intensity field TIFFs and metadata
        sidecars (e.g. an ``OME-TIFF-MIP/`` folder).
    plate_path
        Full path to the shared plate .zarr store.
    plate_name
        Name for the plate, used only if it does not already exist.
    n_batches, batch_id
        PBS-style batch splitting, by well.
    label_dir, feature_csv_dir, point_object_channel_names
        See :func:`convert_tiff_well_to_ome_ngff`.
    y_direction, x_direction, placement
        See :func:`get_field_layout_from_tiff_metadata`.
    channel_names
        List of channel names in case those found in the TIFF metadata are
        incorrect.
    """
    in_path = Path(in_path)
    plate_path = Path(plate_path)

    ensure_plate_exists(plate_path, plate_name or plate_path.stem)

    nd2_stems = sorted(p.stem[: -len("_metadata")] for p in in_path.glob("*_metadata.csv"))
    n_wells_per_batch = -(-len(nd2_stems) // n_batches)
    batch_stems = nd2_stems[batch_id * n_wells_per_batch : (batch_id + 1) * n_wells_per_batch]

    logger.info(f"Converting TIFF wells to OME-NGFF: {batch_stems}")
    for nd2_stem in batch_stems:
        convert_tiff_well_to_ome_ngff(
            nd2_stem=nd2_stem,
            tiff_dir=in_path,
            plate_path=plate_path,
            label_dir=label_dir,
            feature_csv_dir=feature_csv_dir,
            point_object_channel_names=point_object_channel_names,
            y_direction=y_direction,
            x_direction=x_direction,
            placement=placement,
            channel_names=channel_names,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="tiff_to_ome_ngff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert a folder of field TIFFs to OME-NGFF")
    convert_parser.add_argument(
        "-i", "--in_path", help="directory containing field TIFFs and metadata sidecars", required=True
    )
    convert_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    convert_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    convert_parser.add_argument(
        "--batch",
        nargs=2,
        type=int,
        default=[1, 0],
        metavar=("N_BATCHES", "BATCH_ID"),
        help="number of batches and the current batch to process (0-indexed)",
    )
    convert_parser.add_argument(
        "-y", "--y_direction", default="down", help='direction of increasing (stage) y-coordinates ("up" or "down")'
    )
    convert_parser.add_argument(
        "-x", "--x_direction", default="left", help='direction of increasing (stage) x-coordinates ("left" or "right")'
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
    convert_parser.add_argument(
        "-l",
        "--label_dir",
        default=None,
        help="directory containing one (possibly multi-channel) label TIFF per field",
    )
    convert_parser.add_argument(
        "-f",
        "--feature_csv_dir",
        default=None,
        help="directory containing one already-aggregated quantify() CSV per field",
    )
    convert_parser.add_argument(
        "--point_object_channel_names",
        type=str,
        nargs="+",
        default=None,
        help="label channel names, if any, with no stable per-pixel identity (see quantify()'s point_objects)",
    )
    convert_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    ensure_parser = subparsers.add_parser("ensure-plate", help="Idempotently create a plate store if it doesn't exist")
    ensure_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    ensure_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    ensure_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "convert":
        tiff_to_ome_ngff(
            in_path=args.in_path,
            plate_path=args.plate_path,
            plate_name=args.plate_name,
            n_batches=args.batch[0],
            batch_id=args.batch[1],
            label_dir=args.label_dir,
            feature_csv_dir=args.feature_csv_dir,
            point_object_channel_names=args.point_object_channel_names,
            y_direction=args.y_direction,
            x_direction=args.x_direction,
            placement=args.placement,
            channel_names=args.channel_names,
        )
    elif args.command == "ensure-plate":
        ensure_plate_exists(args.plate_path, args.plate_name or Path(args.plate_path).stem)
