"""Assemble a whole-plate OME-NGFF (OME-Zarr) store from an existing
OME-TIFF pipeline: per-field intensity TIFFs (from ``nd2_to_ome_tiff.py``),
per-field segmentation label TIFFs, and per-field ``quantify()`` measurement
CSVs.

Input contract (matches the lab's own established convention -- see
``run_segment_and_quantify.py`` in the ``berrygroup/publications`` repo):
one intensity TIFF, one label TIFF, and one measurements CSV per field, all
sharing the intensity TIFF's own filename stem, each in their own directory
(e.g. ``OME-TIFF-MIP/``, ``SEGMENTATION/``, ``QUANTIFICATION/``).

Real stage positions are read from the metadata sidecar ``nd2_to_ome_tiff.py``
writes alongside the field TIFFs (only when called with ``mip=True`` or
``keep_stacks=True``) -- see ``nd2_parse_metadata.py::nd2_extract_metadata_and_save``.
Placement is "grid" (default) or "exact", same choice and same meaning
as ``nd2_to_ome_ngff.py`` -- see :func:`get_field_layout_from_tiff_metadata`.

Robust to partial upstream failure, field by field: a missing intensity or
label TIFF is substituted with a blank (all-zero) array of that field's tile
shape; a missing measurements CSV simply contributes no rows to that
object's feature table. The metadata sidecar's own field list is
authoritative for "which fields should exist," independent of which
downstream files actually landed on disk -- see :func:`_discover_well_manifest`.
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
    label_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    feature_csv_dirs: Optional[Dict[str, Union[str, Path]]] = None,
) -> pd.DataFrame:
    """Cross-reference the metadata sidecar's field list (authoritative --
    fixed at acquisition time, independent of which downstream files
    actually exist) against what's actually present in the intensity TIFF
    directory, each named label directory, and each named feature-CSV
    directory. Logs a per-source summary naming exactly which field IDs are
    missing, so gaps are observable rather than silently papered over.

    Returns
    -------
    pd.DataFrame
        One row per ``field_id`` (plus ``stage_x_abs``/``stage_y_abs``/
        ``filename_ome_tiff`` from the sidecar), with an added
        ``intensity_exists`` boolean column, and one ``label_exists_{name}``/
        ``feature_exists_{name}`` boolean column per ``label_dirs``/
        ``feature_csv_dirs`` entry.
    """
    label_dirs = label_dirs or {}
    feature_csv_dirs = feature_csv_dirs or {}

    df = _read_metadata_csv(nd2_stem, tiff_dir)
    stems = df["filename_ome_tiff"].apply(lambda f: Path(f).stem)

    df["intensity_exists"] = df["filename_ome_tiff"].apply(lambda f: (Path(tiff_dir) / f).exists())
    missing = df.loc[~df["intensity_exists"], "field_id"].tolist()
    if missing:
        logger.warning(f"Well {nd2_stem}: intensity TIFF missing for field_id(s) {missing}")

    for name, label_dir in label_dirs.items():
        col = f"label_exists_{name}"
        df[col] = df["filename_ome_tiff"].apply(lambda f: (Path(label_dir) / f).exists())
        missing = df.loc[~df[col], "field_id"].tolist()
        if missing:
            logger.warning(f"Well {nd2_stem}: label '{name}' missing for field_id(s) {missing}")

    for name, feature_dir in feature_csv_dirs.items():
        col = f"feature_exists_{name}"
        df[col] = stems.apply(lambda s: (Path(feature_dir) / f"{s}.csv").exists())
        missing = df.loc[~df[col], "field_id"].tolist()
        if missing:
            logger.warning(f"Well {nd2_stem}: feature CSV '{name}' missing for field_id(s) {missing}")

    logger.info(
        f"Well {nd2_stem}: {len(df)} fields expected; "
        f"intensity found for {int(df['intensity_exists'].sum())}/{len(df)}"
    )
    return df


def _channel_is_point_object(
    channel_name: str, feature_csv_dirs: Dict[str, Union[str, Path]], manifest: pd.DataFrame
) -> bool:
    """Determine whether a channel is a point-object channel by reading
    ``is_point_object`` off the first available feature CSV for it.

    Falls back to ``False`` (blob object) with a logged warning if that
    channel has no feature CSV available for any field in this well --
    point/blob-ness genuinely can't be determined from the label file
    alone (see module docstring in ``blimp/ome_ngff/labels.py``).
    """
    feature_dir = feature_csv_dirs.get(channel_name)
    if feature_dir is None:
        return False
    exists_col = f"feature_exists_{channel_name}"
    for filename_ome_tiff in manifest.loc[manifest[exists_col], "filename_ome_tiff"]:
        csv_path = Path(feature_dir) / f"{Path(filename_ome_tiff).stem}.csv"
        df = pd.read_csv(csv_path, nrows=1)
        if "is_point_object" in df.columns:
            return bool(df["is_point_object"].iloc[0])
        break
    logger.warning(
        f"Could not determine whether '{channel_name}' is a point-object channel "
        "(no feature CSV available, or it predates the is_point_object column); "
        "treating it as a regular (blob) object."
    )
    return False


def convert_tiff_well_to_ome_ngff(
    nd2_stem: str,
    tiff_dir: Union[str, Path],
    plate_path: Union[str, Path],
    label_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    feature_csv_dirs: Optional[Dict[str, Union[str, Path]]] = None,
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
    label_dirs
        Maps a label name (e.g. ``"Nuclei"``) to the directory containing
        that channel's per-field label TIFFs (same filenames as the
        intensity TIFFs). Omit a channel entirely to skip writing it. Each
        label TIFF's own Z depth must match ``tiff_dir``'s (both 1 for a
        MIP, or both equal for a real stack) -- ``quantify()`` and the rest
        of ``segment.py`` support 3D label images even though
        ``segment_nuclei_cellpose`` itself only produces 2D ones.
    feature_csv_dirs
        Maps a label name to the directory containing that channel's
        per-field ``quantify()`` measurement CSVs (same filename stems).
        The key names **the label whose `label`/`parent_label` values these
        rows key against** -- the object's own name for a non-aggregated
        ``quantify()`` result, or the *parent's* name when
        ``quantify(aggregate=True)`` was used (the caller decides this by
        which name it passes here, not this function).
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
    label_dirs = label_dirs or {}
    feature_csv_dirs = feature_csv_dirs or {}

    logger.info(f"Reading layout for well {nd2_stem}")
    layout = get_field_layout_from_tiff_metadata(
        nd2_stem, tiff_dir, y_direction=y_direction, x_direction=x_direction, placement=placement
    )
    is_mip = layout.tile_shape[2] == 1
    manifest = _discover_well_manifest(nd2_stem, tiff_dir, label_dirs, feature_csv_dirs)

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

    for label_name, label_dir in label_dirs.items():
        exists_col = f"label_exists_{label_name}"
        is_point_object = _channel_is_point_object(label_name, feature_csv_dirs, manifest)

        # Point objects are read as flat 2D masks (_write_well_points has no
        # 3D support); blob objects keep their real Z depth so a genuine 3D
        # label places one slice per Z-plane rather than broadcasting a 2D
        # mask across every plane.
        field_arrays: Dict[int, Optional[np.ndarray]] = {}
        for field_id in layout.field_ids:
            row = manifest_by_field.loc[field_id]
            if not row[exists_col]:
                field_arrays[field_id] = None
                continue
            path = Path(label_dir) / row["filename_ome_tiff"]
            label_image = BioImage(str(path))
            if is_point_object:
                field_arrays[field_id] = np.squeeze(label_image.get_image_data("YX"))
            else:
                field_arrays[field_id] = label_image.get_image_data("ZYX", C=0, T=0)

        field_features: Dict[int, Optional[pd.DataFrame]] = {}
        feature_dir = feature_csv_dirs.get(label_name)
        if feature_dir is not None:
            exists_col_feat = f"feature_exists_{label_name}"
            for field_id in layout.field_ids:
                row = manifest_by_field.loc[field_id]
                if not row[exists_col_feat]:
                    field_features[field_id] = None
                    continue
                stem = Path(row["filename_ome_tiff"]).stem
                field_features[field_id] = pd.read_csv(Path(feature_dir) / f"{stem}.csv")
        else:
            field_features = {field_id: None for field_id in layout.field_ids}

        well_name = f"{layout.row}{layout.column:02d}"
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
            if feature_dir is not None:
                _write_well_features(
                    container=container, label_name=label_name, field_dataframes=field_features, well_name=well_name
                )


def tiff_to_ome_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    plate_name: Optional[str] = None,
    n_batches: int = 1,
    batch_id: int = 0,
    label_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    feature_csv_dirs: Optional[Dict[str, Union[str, Path]]] = None,
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
    label_dirs, feature_csv_dirs
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
            label_dirs=label_dirs,
            feature_csv_dirs=feature_csv_dirs,
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
            y_direction=args.y_direction,
            x_direction=args.x_direction,
            placement=args.placement,
            channel_names=args.channel_names,
        )
    elif args.command == "ensure-plate":
        ensure_plate_exists(args.plate_path, args.plate_name or Path(args.plate_path).stem)
