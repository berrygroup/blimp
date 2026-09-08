"""Tests for the shared blimp.ome_ngff writer core (layout, metadata,
plate/well registration, label/point placement, feature tables), used by
every source-format writer (nd2, TIFF-pipeline)."""
from typing import Any, Dict
from pathlib import Path
import time
import logging

from ngio.ome_zarr_meta.ngio_specs import PixelSize
import ngio
import zarr
import numpy as np
import pandas as pd
import pytest

from blimp.ome_ngff.plate import (
    open_well_image,
    resolve_plate_path,
    build_plate_pyramid,
    ensure_plate_exists,
    build_feature_pyramid,
    _read_plate_wide_features,
    _discover_wells_with_label,
)
from blimp.ome_ngff.labels import (
    global_id,
    _offset_label_ids,
    well_label_offset,
    _write_well_labels,
    _write_well_points,
    MAX_OBJECTS_PER_FIELD,
    WELL_LABEL_OFFSET_STEP,
    _validate_field_offset_capacity,
)
from blimp.ome_ngff.layout import (
    FieldLayout,
    _WELL_NAME_RE,
    _parse_well_name,
    _cluster_grid_index,
    _exact_pixel_offset,
    _build_fov_roi_table,
)
from blimp.ome_ngff.features import _write_well_features, _offset_feature_table_ids
from blimp.ome_ngff.metadata import _downsample_yx, _build_ngff_v05_metadata
from blimp.preprocessing.tiff_to_ome_ngff import convert_tiff_well_to_ome_ngff

# --------------------------------------------------------------------------- #
# layout.py -- field-of-view layout and grid-clustering math
# --------------------------------------------------------------------------- #


def test_well_name_re_parses_row_and_column():
    match = _WELL_NAME_RE.match("C09_0000")
    assert match is not None
    assert match.group(1) == "C"
    assert match.group(2) == "09"


def test_well_name_re_handles_multi_letter_rows():
    match = _WELL_NAME_RE.match("AA12")
    assert match is not None
    assert match.group(1) == "AA"
    assert match.group(2) == "12"


def test_parse_well_name_prefers_position_name():
    row, column = _parse_well_name("/some/path/unrelated_filename.nd2", ["C09_0000", "C09_0001"])
    assert (row, column) == ("C", 9)


def test_parse_well_name_falls_back_to_filename():
    row, column = _parse_well_name("/some/path/WellB02_Channel647_Seq0001.nd2", [None])
    assert (row, column) == ("B", 2)


def test_parse_well_name_raises_when_neither_source_has_a_well():
    with pytest.raises(ValueError, match="Could not determine a well identifier"):
        _parse_well_name("/some/path/221103_brightfield_488_568_647_1.nd2", [None])


def test_cluster_grid_index_snaps_jittered_positions_to_a_clean_grid():
    """Real stage positions from a 2x2 acquisition: nominal 746 um tile
    pitch, with ~28 um of jitter/axis cross-talk on the perpendicular axis
    -- clustering should collapse that jitter to exactly two clusters per
    axis, not four."""
    stage_x = np.array([15614.9, 14868.8, 14896.2, 15642.5])
    col_idx = _cluster_grid_index(stage_x, tile_extent=746.9)
    assert set(col_idx.tolist()) == {0, 1}
    # the two smallest x values (pos1, pos2) must share a cluster, distinct
    # from the two largest (pos0, pos3)
    assert col_idx[1] == col_idx[2]
    assert col_idx[0] == col_idx[3]
    assert col_idx[0] != col_idx[1]


def test_cluster_grid_index_single_cluster():
    values = np.array([100.0, 101.0, 99.5, 100.5])
    idx = _cluster_grid_index(values, tile_extent=50.0)
    assert set(idx.tolist()) == {0}


def test_cluster_grid_index_orders_clusters_ascending():
    values = np.array([1000.0, 0.0, 500.0])
    idx = _cluster_grid_index(values, tile_extent=100.0)
    # cluster id must increase monotonically with the underlying value
    assert idx[1] < idx[2] < idx[0]


def test_exact_pixel_offset_maps_min_stage_value_to_zero():
    stage = np.array([100.0, 150.0, 125.0])
    offset = _exact_pixel_offset(stage, pixel_size=5.0, reverse=False)
    assert offset.tolist() == [0, 10, 5]


def test_exact_pixel_offset_reverse_maps_max_stage_value_to_zero():
    stage = np.array([100.0, 150.0, 125.0])
    offset = _exact_pixel_offset(stage, pixel_size=5.0, reverse=True)
    assert offset.tolist() == [10, 0, 5]


def _make_field_layout(**overrides: Any) -> FieldLayout:
    defaults: Dict[str, Any] = dict(
        row="C",
        column=9,
        offsets=[(0, 0), (0, 100), (100, 0), (100, 100)],
        tile_shape=(1, 2, 3, 100, 100),
        canvas_shape=(1, 2, 3, 200, 200),
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        pixel_size_z=1.0,
        channel_names=["DAPI", "GFP"],
        channel_colors=["0000FF", "00FF00"],
        position_names=["C09_0000", "C09_0001", "C09_0002", "C09_0003"],
    )
    defaults.update(overrides)
    return FieldLayout(**defaults)


def test_field_layout_defaults_field_ids_to_1_indexed_enumeration():
    layout = _make_field_layout()
    assert layout.field_ids == [1, 2, 3, 4]


def test_field_layout_accepts_explicit_field_ids():
    layout = _make_field_layout(field_ids=[7, 12, 3, 9])
    assert layout.field_ids == [7, 12, 3, 9]


def test_build_fov_roi_table_round_trips_pixel_offsets():
    layout = _make_field_layout()
    table = _build_fov_roi_table(layout)
    rois = table.rois()
    assert [r.name for r in rois] == layout.position_names

    pixel_size = PixelSize(x=layout.pixel_size_x, y=layout.pixel_size_y, z=layout.pixel_size_z)
    for roi, (y0, x0) in zip(rois, layout.offsets):
        pixel_slices = roi.to_slicing_dict(pixel_size=pixel_size)
        assert pixel_slices["y"] == slice(y0, y0 + layout.tile_shape[3])
        assert pixel_slices["x"] == slice(x0, x0 + layout.tile_shape[4])


def test_build_fov_roi_table_falls_back_to_generic_name_when_position_name_missing():
    layout = _make_field_layout(offsets=[(0, 0)], position_names=[None])
    table = _build_fov_roi_table(layout)
    assert [r.name for r in table.rois()] == ["FOV_0"]


def test_build_fov_roi_table_falls_back_to_unit_pixel_size_and_warns(caplog):
    layout = _make_field_layout(
        offsets=[(0, 0)],
        position_names=[None],
        pixel_size_x=0.0,
        pixel_size_y=None,
    )
    with caplog.at_level(logging.WARNING):
        table = _build_fov_roi_table(layout)
    assert "falling back to" in caplog.text

    roi = table.rois()[0]
    pixel_slices = roi.to_slicing_dict(pixel_size=PixelSize(x=1.0, y=1.0, z=1.0))
    assert pixel_slices["y"] == slice(0, layout.tile_shape[3])
    assert pixel_slices["x"] == slice(0, layout.tile_shape[4])


# --------------------------------------------------------------------------- #
# metadata.py -- NGFF 0.5 metadata construction and pyramid downsampling
# --------------------------------------------------------------------------- #


def test_downsample_yx_halves_spatial_dims_only():
    arr = np.arange(1 * 2 * 3 * 8 * 8, dtype=np.uint16).reshape(1, 2, 3, 8, 8)
    out = _downsample_yx(arr)
    assert out.shape == (1, 2, 3, 4, 4)


def test_downsample_yx_is_nearest_neighbor_striding():
    arr = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
    arr[0, 0, 0] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    out = _downsample_yx(arr)
    np.testing.assert_array_equal(out[0, 0, 0], [[1, 3], [9, 11]])


def test_build_ngff_v05_metadata_structure():
    meta = _build_ngff_v05_metadata(
        image_name="C09",
        num_levels=3,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        pixel_size_z=1.0,
        channel_names=["DAPI", "GFP"],
        channel_colors=["0000FF", "00FF00"],
    )
    ome = meta["ome"]
    assert ome["version"] == "0.5"

    multiscale = ome["multiscales"][0]
    assert multiscale["name"] == "C09"
    assert [a["name"] for a in multiscale["axes"]] == ["t", "c", "z", "y", "x"]

    datasets = multiscale["datasets"]
    assert [d["path"] for d in datasets] == ["0", "1", "2"]
    # Level 0 scale is the raw pixel size; each subsequent level doubles it.
    assert datasets[0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 0.5, 0.5]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert datasets[2]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 2.0, 2.0]

    channels = ome["omero"]["channels"]
    assert [c["label"] for c in channels] == ["DAPI", "GFP"]
    assert [c["color"] for c in channels] == ["0000FF", "00FF00"]


# --------------------------------------------------------------------------- #
# plate.py -- shared OME-Zarr plate/well registration and image writing
# --------------------------------------------------------------------------- #


def test_open_well_image_raises_clear_error_for_missing_kind(tmp_path):
    """No multi-position .nd2 fixture exists to exercise the full write
    pipeline end-to-end, so this builds just enough of a well's on-disk
    structure by hand -- a "mip" image directory plus the well-level
    ``ome.well.images`` metadata a real conversion would produce -- to
    exercise open_well_image's validation in isolation."""
    plate_path = tmp_path / "plate.zarr"
    well_relative_path = "C/09"
    well_path = plate_path / well_relative_path
    (well_path / "mip").mkdir(parents=True)

    well_group = zarr.open_group(str(well_path), mode="a", zarr_format=3)
    well_group.attrs["ome"] = {
        "version": "0.5",
        "well": {"version": "0.5", "images": [{"path": "mip"}]},
    }

    with pytest.raises(FileNotFoundError) as excinfo:
        open_well_image(plate_path, well_relative_path, kind="stack")
    message = str(excinfo.value)
    assert "'stack'" in message
    assert "'mip'" in message
    assert "keep_stacks=True" in message


def test_resolve_plate_path_uses_dot_zarr_path_as_is():
    assert resolve_plate_path("/some/experiment.zarr") == Path("/some/experiment.zarr")


def test_resolve_plate_path_appends_plate_dot_zarr_to_a_bare_folder():
    assert resolve_plate_path("/some/experiment") == Path("/some/experiment/plate.zarr")


def test_ensure_plate_exists_predeclares_full_384_well_grid(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    plate = ensure_plate_exists(plate_path, "test_plate")
    assert plate.rows == [chr(ord("A") + i) for i in range(16)]
    assert plate.columns == [f"{i:02d}" for i in range(1, 25)]

    # pre-declaring the grid costs nothing in storage: nothing exists yet
    # besides the plate's own zarr.json until a well is actually written.
    assert [p.name for p in plate_path.iterdir()] == ["zarr.json"]


def test_ensure_plate_exists_predeclares_96_well_grid_when_requested(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    plate = ensure_plate_exists(plate_path, "test_plate", plate_size="96")
    assert plate.rows == [chr(ord("A") + i) for i in range(8)]
    assert plate.columns == [f"{i:02d}" for i in range(1, 13)]


def test_ensure_plate_exists_is_idempotent(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    plate2 = ensure_plate_exists(plate_path, "test_plate")
    assert plate2.rows == [chr(ord("A") + i) for i in range(16)]


def test_ensure_plate_exists_raises_clear_error_for_non_empty_non_plate_directory(tmp_path):
    """A plate_path that already exists and has content, but isn't itself a
    valid OME-Zarr plate store (e.g. -o . run from an ordinary working
    directory), must raise one clear FileExistsError -- not the confusing
    chained NgioFileExistsError/NgioFileNotFoundError pair that would
    otherwise surface: create_empty_plate refuses because the directory is
    non-empty, and the fallback open then fails because there's no real
    plate store there either."""
    (tmp_path / "some_unrelated_file.txt").write_text("hello")
    with pytest.raises(FileExistsError, match="valid OME-Zarr plate store"):
        ensure_plate_exists(tmp_path, "test_plate")


def _write_intensity_tiff(path: Path, fill_value: int) -> None:
    """A minimal real single-field OME-TIFF filled with a recognizable
    constant, so a real well's data can be told apart from an untouched
    (all-zero) declared grid position -- unlike a genuinely blank TIFF."""
    from bioio_base.types import PhysicalPixelSizes
    from bioio_ome_tiff.writers import OmeTiffWriter

    OmeTiffWriter.save(
        data=np.full((1, 2, 1, 16, 16), fill_value, dtype="uint16"),
        uri=str(path),
        dim_order="TCZYX",
        channel_names=["DAPI", "GFP"],
        physical_pixel_sizes=PhysicalPixelSizes(1.0, 0.5, 0.5),
    )


def _write_label_tiff_single_channel(path: Path, fill_value: int) -> None:
    from bioio_base.types import PhysicalPixelSizes
    from bioio_ome_tiff.writers import OmeTiffWriter

    OmeTiffWriter.save(
        data=np.full((1, 1, 1, 16, 16), fill_value, dtype="uint16"),
        uri=str(path),
        dim_order="TCZYX",
        channel_names=["Nuclei"],
        physical_pixel_sizes=PhysicalPixelSizes(1.0, 0.5, 0.5),
    )


def _write_one_well(
    tmp_path: Path,
    plate_path: Path,
    nd2_stem: str,
    fill_value: int,
    with_label: bool = False,
    num_levels: int = 2,
    feature_value: Any = None,
    feature_name: str = "Nuclei_area",
) -> None:
    """Write one real well into an already-``ensure_plate_exists``-created
    plate, via the same public ``convert_tiff_well_to_ome_ngff`` pipeline a
    real TIFF conversion run uses -- exercises ``build_plate_pyramid``
    against a genuine multi-well plate store, not a hand-built stand-in.

    ``feature_value`` (only meaningful together with ``with_label=True``)
    additionally writes a one-row features CSV for this well's single
    object (raw local label ``fill_value``), so ``build_feature_pyramid``/
    ``_read_plate_wide_features`` have a real ``Nuclei_features`` table to
    read -- ``feature_name`` names that CSV's own measurement column."""
    tiff_dir = tmp_path / nd2_stem / "intensity"
    tiff_dir.mkdir(parents=True)
    filename = f"{nd2_stem}_0001.ome.tiff"
    _write_intensity_tiff(tiff_dir / filename, fill_value)
    pd.DataFrame([{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}]).to_csv(
        tiff_dir / f"{nd2_stem}_metadata.csv", index=False
    )

    label_dir = None
    if with_label:
        label_dir = tmp_path / nd2_stem / "labels"
        label_dir.mkdir(parents=True)
        _write_label_tiff_single_channel(label_dir / filename, fill_value)

    feature_csv_dir = None
    if feature_value is not None:
        feature_csv_dir = tmp_path / nd2_stem / "features"
        feature_csv_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "label": [fill_value],
                "parent_label_name": ["Nuclei"],
                "is_point_object": [False],
                feature_name: [feature_value],
            }
        ).to_csv(feature_csv_dir / f"{Path(filename).stem}.csv", index=False)

    convert_tiff_well_to_ome_ngff(
        nd2_stem=nd2_stem,
        tiff_dir=tiff_dir,
        plate_path=plate_path,
        label_dir=label_dir,
        feature_csv_dir=feature_csv_dir,
        num_levels=num_levels,
    )


def test_build_plate_pyramid_places_real_wells_at_their_grid_position(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=7)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=42)

    pyramid = build_plate_pyramid(plate_path, kind="mip")
    assert len(pyramid) == 2  # num_levels passed to _write_one_well

    level0 = pyramid[0]
    tile = 16
    pitch = round(tile * 1.05)
    n_rows, n_cols = 16, 24  # ensure_plate_exists' default "384" grid
    # (channels, z=1, y, x) -- a mip still keeps its own size-1 Z axis.
    assert level0.shape == (2, 1, n_rows * pitch, n_cols * pitch)

    def _well_slice(row: str, column: int):
        row_idx, col_idx = ord(row) - ord("A"), column - 1
        y0, x0 = row_idx * pitch, col_idx * pitch
        return level0[:, 0, y0 : y0 + tile, x0 : x0 + tile].compute()

    np.testing.assert_array_equal(_well_slice("C", 9), np.full((2, tile, tile), 7, dtype=level0.dtype))
    np.testing.assert_array_equal(_well_slice("F", 14), np.full((2, tile, tile), 42, dtype=level0.dtype))

    # An empty declared grid position (never written) stays exactly zero.
    assert np.all(level0[:, 0, 0:5, 0:5].compute() == 0)
    # So does the gap margin just past a real well's own tile, within its
    # own pitch cell -- the whole point of the gap.
    row_idx, col_idx = ord("C") - ord("A"), 9 - 1
    y0, x0 = row_idx * pitch, col_idx * pitch
    assert np.all(level0[:, 0, y0 + tile : y0 + pitch, x0 : x0 + pitch].compute() == 0)


def test_build_plate_pyramid_label_name_variant(tmp_path):
    """C09 and F14 are given the *same* fill_value, so their raw within-well
    global IDs collide -- build_plate_pyramid must still tell them apart by
    applying each well's own well_label_offset. A third well with no label
    at all (K16) must still be skipped (its grid position stays zero), not
    raise."""
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=3, with_label=True)
    _write_one_well(tmp_path, plate_path, "WellK16_Seq0001", fill_value=9, with_label=False)

    pyramid = build_plate_pyramid(plate_path, kind="mip", label_name="Nuclei")
    level0 = pyramid[0]
    assert level0.dtype == np.int64
    # (z=1, y, x) -- no channel axis for a label.
    assert level0.ndim == 3

    tile = 16
    pitch = round(tile * 1.05)
    # _write_well_labels offsets local label IDs into a global,
    # field-keyed range (global_id = field_id * MAX_OBJECTS_PER_FIELD +
    # local_id) -- our single field is field_id 1, so the fill_value=3
    # pixels land here, not as the raw local ID. Both wells share this same
    # raw value -- only well_label_offset tells them apart at plate scale.
    raw_global_id = 1 * MAX_OBJECTS_PER_FIELD + 3

    def _well_slice(row: str, column: int):
        row_idx, col_idx = ord(row) - ord("A"), column - 1
        y0, x0 = row_idx * pitch, col_idx * pitch
        return level0[0, y0 : y0 + tile, x0 : x0 + tile].compute()

    c09_offset = well_label_offset(ord("C") - ord("A"), 9 - 1)
    f14_offset = well_label_offset(ord("F") - ord("A"), 14 - 1)
    assert c09_offset != f14_offset  # the offsets themselves must differ...
    np.testing.assert_array_equal(_well_slice("C", 9), np.full((tile, tile), c09_offset + raw_global_id))
    np.testing.assert_array_equal(_well_slice("F", 14), np.full((tile, tile), f14_offset + raw_global_id))
    # ...so despite an identical raw ID, the two wells never collide.
    assert not np.array_equal(_well_slice("C", 9), _well_slice("F", 14))

    assert np.all(_well_slice("K", 16) == 0)


def test_build_plate_pyramid_raises_when_no_well_has_the_requested_label(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=7)

    with pytest.raises(ValueError, match="Nonexistent"):
        build_plate_pyramid(plate_path, kind="mip", label_name="Nonexistent")


def test_build_plate_pyramid_construction_time_does_not_scale_with_declared_grid_size(tmp_path):
    """Construction cost must track the number of *real* wells, not the
    plate's declared row/column grid size -- unlike
    ``ome_zarr.reader.Plate.get_stitched_grid``, which eagerly builds a
    per-declared-grid-cell array-concatenation graph."""
    small_plate_path = tmp_path / "small_plate.zarr"
    ensure_plate_exists(small_plate_path, "small_plate", plate_size="96")
    _write_one_well(tmp_path / "small", small_plate_path, "WellC09_Seq0001", fill_value=7)

    large_plate_path = tmp_path / "large_plate.zarr"
    ensure_plate_exists(large_plate_path, "large_plate", plate_size="384")
    _write_one_well(tmp_path / "large", large_plate_path, "WellC09_Seq0001", fill_value=7)

    t0 = time.time()
    build_plate_pyramid(small_plate_path, kind="mip")
    small_grid_time = time.time() - t0

    t0 = time.time()
    build_plate_pyramid(large_plate_path, kind="mip")
    large_grid_time = time.time() - t0

    # "384" declares 4x as many grid cells as "96" (16x24 vs 8x12) with the
    # same single real well -- a generous multiple (not 1x) to absorb
    # timing noise, but nowhere near proportional to declared grid size.
    assert large_grid_time < max(small_grid_time * 4, 1.0)


def test_read_plate_wide_features_returns_none_when_no_matching_table(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True)  # no feature_value

    assert _read_plate_wide_features(plate_path, "Nuclei", kind="mip") is None


def test_read_plate_wide_features_merges_wells_with_correct_offsets(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    # Same raw local label (3) in both wells -- only each well's own offset
    # tells their global_id_numeric apart.
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=3, with_label=True, feature_value=222.0)

    df = _read_plate_wide_features(plate_path, "Nuclei", kind="mip")
    assert df is not None
    assert len(df) == 2
    assert set(df["Nuclei_area"].tolist()) == {111.0, 222.0}

    raw_global_id = 1 * MAX_OBJECTS_PER_FIELD + 3
    c09_offset = well_label_offset(ord("C") - ord("A"), 9 - 1)
    f14_offset = well_label_offset(ord("F") - ord("A"), 14 - 1)
    expected = {c09_offset + raw_global_id: 111.0, f14_offset + raw_global_id: 222.0}
    assert dict(zip(df["label"], df["Nuclei_area"])) == expected


def test_build_feature_pyramid_shows_correct_value_and_nan_elsewhere(tmp_path):
    """C09 and F14 share the same raw local label (3), so only each well's
    own well_label_offset -- applied identically to the pixel array and to
    the features table's "label" column -- keeps their heatmap values from
    bleeding into each other. K16 has the label but no matching feature row
    at all (no feature_csv_dir given), so its region must read back NaN
    despite having real label pixels."""
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=3, with_label=True, feature_value=222.0)
    _write_one_well(tmp_path, plate_path, "WellK16_Seq0001", fill_value=3, with_label=True)  # no feature_value

    pyramid = build_feature_pyramid(plate_path, "Nuclei", "Nuclei_area", kind="mip")
    level0 = pyramid[0]
    assert level0.dtype == np.float32
    assert level0.ndim == 3  # (z=1, y, x), matching build_plate_pyramid's own label variant

    tile = 16
    pitch = round(tile * 1.05)

    def _well_slice(row: str, column: int):
        row_idx, col_idx = ord(row) - ord("A"), column - 1
        y0, x0 = row_idx * pitch, col_idx * pitch
        return level0[0, y0 : y0 + tile, x0 : x0 + tile].compute()

    np.testing.assert_array_equal(_well_slice("C", 9), np.full((tile, tile), 111.0, dtype=np.float32))
    np.testing.assert_array_equal(_well_slice("F", 14), np.full((tile, tile), 222.0, dtype=np.float32))
    assert np.all(np.isnan(_well_slice("K", 16)))  # labeled, but no matching feature row
    assert np.all(np.isnan(level0[0, 0:5, 0:5].compute()))  # never-written declared grid position


def test_build_feature_pyramid_raises_when_no_well_has_the_requested_table(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True)  # no feature_value

    with pytest.raises(ValueError, match="Nuclei_features"):
        build_feature_pyramid(plate_path, "Nuclei", "Nuclei_area", kind="mip")


def test_build_feature_pyramid_raises_when_feature_name_is_not_a_column(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)

    with pytest.raises(ValueError, match="Nonexistent_feature"):
        build_feature_pyramid(plate_path, "Nuclei", "Nonexistent_feature", kind="mip")


def test_build_feature_pyramid_construction_time_does_not_scale_with_declared_grid_size(tmp_path):
    small_plate_path = tmp_path / "small_plate.zarr"
    ensure_plate_exists(small_plate_path, "small_plate", plate_size="96")
    _write_one_well(
        tmp_path / "small", small_plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0
    )

    large_plate_path = tmp_path / "large_plate.zarr"
    ensure_plate_exists(large_plate_path, "large_plate", plate_size="384")
    _write_one_well(
        tmp_path / "large", large_plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0
    )

    t0 = time.time()
    build_feature_pyramid(small_plate_path, "Nuclei", "Nuclei_area", kind="mip")
    small_grid_time = time.time() - t0

    t0 = time.time()
    build_feature_pyramid(large_plate_path, "Nuclei", "Nuclei_area", kind="mip")
    large_grid_time = time.time() - t0

    assert large_grid_time < max(small_grid_time * 4, 1.0)


def test_build_feature_pyramid_wells_filter_restricts_to_a_single_well(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=3, with_label=True, feature_value=222.0)

    pyramid = build_feature_pyramid(plate_path, "Nuclei", "Nuclei_area", kind="mip", wells="C/09")
    level0 = pyramid[0]
    tile = 16
    pitch = round(tile * 1.05)

    def _well_slice(row: str, column: int):
        row_idx, col_idx = ord(row) - ord("A"), column - 1
        y0, x0 = row_idx * pitch, col_idx * pitch
        return level0[0, y0 : y0 + tile, x0 : x0 + tile].compute()

    np.testing.assert_array_equal(_well_slice("C", 9), np.full((tile, tile), 111.0, dtype=np.float32))
    # F14 has real data on disk, but is outside the `wells` filter -- its
    # region must read back NaN, exactly like a never-written grid position.
    assert np.all(np.isnan(_well_slice("F", 14)))


def test_build_feature_pyramid_wells_filter_accepts_a_list(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)
    _write_one_well(tmp_path, plate_path, "WellF14_Seq0001", fill_value=3, with_label=True, feature_value=222.0)
    _write_one_well(tmp_path, plate_path, "WellK16_Seq0001", fill_value=3, with_label=True, feature_value=333.0)

    pyramid = build_feature_pyramid(plate_path, "Nuclei", "Nuclei_area", kind="mip", wells=["C/09", "F/14"])
    level0 = pyramid[0]
    tile = 16
    pitch = round(tile * 1.05)

    def _well_slice(row: str, column: int):
        row_idx, col_idx = ord(row) - ord("A"), column - 1
        y0, x0 = row_idx * pitch, col_idx * pitch
        return level0[0, y0 : y0 + tile, x0 : x0 + tile].compute()

    np.testing.assert_array_equal(_well_slice("C", 9), np.full((tile, tile), 111.0, dtype=np.float32))
    np.testing.assert_array_equal(_well_slice("F", 14), np.full((tile, tile), 222.0, dtype=np.float32))
    assert np.all(np.isnan(_well_slice("K", 16)))


def test_discover_wells_with_label_raises_for_a_well_not_in_the_plate(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    _write_one_well(tmp_path, plate_path, "WellC09_Seq0001", fill_value=3, with_label=True, feature_value=111.0)

    plate = ngio.open_ome_zarr_plate(store=str(plate_path), mode="r")
    with pytest.raises(ValueError, match="Z/99"):
        _discover_wells_with_label(plate, "mip", "Nuclei", wells="Z/99")


# --------------------------------------------------------------------------- #
# labels.py -- segmentation label placement with reproducible global IDs,
# and the point-object GenericRoiTable path
# --------------------------------------------------------------------------- #


def _make_well_layout(**overrides):
    defaults = dict(
        row="C",
        column=9,
        offsets=[(0, 0), (0, 16)],
        tile_shape=(1, 1, 1, 16, 16),
        canvas_shape=(1, 1, 1, 16, 32),
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        pixel_size_z=1.0,
        channel_names=["DAPI"],
        channel_colors=["0000FF"],
        position_names=["C09_0001", "C09_0002"],
        field_ids=[1, 2],
    )
    defaults.update(overrides)
    return FieldLayout(**defaults)


def _make_well_container(tmp_path: Path, canvas_shape=(1, 1, 1, 16, 32)):
    store = str(tmp_path / "plate.zarr")
    return ngio.create_ome_zarr_from_array(
        store=store,
        array=np.zeros(canvas_shape, dtype=np.uint16),
        pixelsize=0.5,
        axes_names=["t", "c", "z", "y", "x"],
        levels=2,
        ngff_version="0.5",
    )


def test_offset_label_ids_uses_field_id_formula():
    local = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    out = _offset_label_ids(local, field_id=3)
    expected_offset = 3 * MAX_OBJECTS_PER_FIELD
    np.testing.assert_array_equal(out, [[0, 1 + expected_offset], [2 + expected_offset, 0]])


def test_offset_label_ids_raises_when_local_id_exceeds_max_objects_per_field():
    local = np.array([MAX_OBJECTS_PER_FIELD], dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds"):
        _offset_label_ids(local, field_id=1, max_objects_per_field=MAX_OBJECTS_PER_FIELD)


def test_global_id_format():
    assert global_id("C09", 4, 123) == "C09_0004_0000123"


def test_validate_field_offset_capacity_passes_for_a_safe_field_id():
    _validate_field_offset_capacity(field_id=1, max_objects_per_field=MAX_OBJECTS_PER_FIELD)


def test_validate_field_offset_capacity_raises_when_range_exceeds_well_capacity():
    huge_field_id = WELL_LABEL_OFFSET_STEP // MAX_OBJECTS_PER_FIELD
    with pytest.raises(ValueError, match="global_id range"):
        _validate_field_offset_capacity(field_id=huge_field_id, max_objects_per_field=MAX_OBJECTS_PER_FIELD)


def test_well_label_offset_is_zero_at_the_origin():
    assert well_label_offset(0, 0) == 0


def test_well_label_offset_uses_default_24_columns():
    assert well_label_offset(1, 0) == WELL_LABEL_OFFSET_STEP * 24
    assert well_label_offset(0, 1) == WELL_LABEL_OFFSET_STEP


def test_well_label_offset_n_cols_is_configurable():
    assert well_label_offset(1, 0, n_cols=12) == WELL_LABEL_OFFSET_STEP * 12


def test_write_well_labels_offsets_and_stitches_two_fields(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    field0[10:14, 10:14] = 2

    field1 = np.zeros((16, 16), dtype=np.uint16)
    field1[2:6, 2:6] = 1  # locally colliding id with field0 -- a different object

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: field1})

    readback = container.get_label("Nuclei").get_as_numpy()
    assert set(np.unique(readback).tolist()) == {
        0,
        1 * MAX_OBJECTS_PER_FIELD + 1,
        1 * MAX_OBJECTS_PER_FIELD + 2,
        2 * MAX_OBJECTS_PER_FIELD + 1,
    }


def test_write_well_labels_consolidates_coarser_pyramid_levels(tmp_path):
    """set_array only ever writes level 0 -- without an explicit
    consolidate() call afterward, every coarser pyramid level is left as
    the empty array derive_label pre-allocated, so the label would
    silently vanish in any viewer that renders from a coarser level when
    zoomed out."""
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: None})

    zarr_group = zarr.open_group(str(tmp_path / "plate.zarr"), mode="r")
    level_1 = zarr_group["labels/Nuclei/1"][:]
    assert set(np.unique(level_1).tolist()) == {0, 1 * MAX_OBJECTS_PER_FIELD + 1}


def test_write_well_labels_blank_substitutes_missing_field(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: None})

    readback = container.get_label("Nuclei").get_as_numpy()
    assert set(np.unique(readback).tolist()) == {0, 1 * MAX_OBJECTS_PER_FIELD + 1}
    # field 2's whole region must stay zero -- confirms it wasn't accidentally
    # populated with stale/garbage data
    assert not readback[..., :, 16:].any()


def test_write_well_labels_places_real_3d_data_per_z_plane_not_broadcast(tmp_path):
    """quantify() and the rest of segment.py support 3D label images (e.g.
    from cellpose's own do_3D=True mode), not just segment_nuclei_cellpose's
    own 2D output -- a genuinely 3D local array must be placed one real
    slice per Z-plane, not broadcast (replicated) across every Z-plane."""
    container = _make_well_container(tmp_path, canvas_shape=(1, 1, 3, 16, 32))
    layout = _make_well_layout(tile_shape=(1, 1, 3, 16, 16), canvas_shape=(1, 1, 3, 16, 32))

    # different object at a different location on each Z-plane
    field0 = np.zeros((3, 16, 16), dtype=np.uint16)
    field0[0, 2:6, 2:6] = 1
    field0[2, 10:14, 10:14] = 2

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: None})

    readback = container.get_label("Nuclei").get_as_numpy()
    assert readback.shape == (1, 3, 16, 32)
    assert set(np.unique(readback[:, 0]).tolist()) == {0, 1 * MAX_OBJECTS_PER_FIELD + 1}
    assert set(np.unique(readback[:, 1]).tolist()) == {0}
    assert set(np.unique(readback[:, 2]).tolist()) == {0, 1 * MAX_OBJECTS_PER_FIELD + 2}


def test_write_well_points_recovers_coordinates_and_merges_features(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    mask0 = np.zeros((16, 16), dtype=np.uint16)
    mask0[3, 3] = 1
    mask0[10, 10] = 1
    features0 = pd.DataFrame({"label": [1, 2], "Spots_area": [1, 1]})

    mask1 = np.zeros((16, 16), dtype=np.uint16)
    mask1[5, 5] = 1
    features1 = pd.DataFrame({"label": [1], "Spots_area": [1]})

    _write_well_points(
        container=container,
        layout=layout,
        well_name="C09",
        channel_name="Spots",
        field_masks={1: mask0, 2: mask1},
        field_features={1: features0, 2: features1},
    )

    table = container.get_generic_roi_table("Spots")
    df = table.dataframe.reset_index()
    assert len(df) == 3
    assert set(df["label"]) == {
        1 * MAX_OBJECTS_PER_FIELD + 1,
        1 * MAX_OBJECTS_PER_FIELD + 2,
        2 * MAX_OBJECTS_PER_FIELD + 1,
    }
    assert set(df["FieldIndex"]) == {"C09_0001_0000001", "C09_0001_0000002", "C09_0002_0000001"}
    assert "Spots_area" in df.columns

    # the field-1 point at mask0[3, 3] should resolve to world (1.5, 1.5) at
    # pixel_size=0.5 -- confirms the field offset was applied correctly
    pixel_size = container.get_image().pixel_size
    rois_by_name = {r.name: r for r in table.rois()}
    roi = rois_by_name["C09_0001_0000001"]
    slices = roi.to_slicing_dict(pixel_size=pixel_size)
    assert slices["y"] == slice(3, 4)
    assert slices["x"] == slice(3, 4)

    # the field-2 point at mask1[5, 5], offset by x0=16, should resolve to
    # pixel x=21
    roi2 = rois_by_name["C09_0002_0000001"]
    slices2 = roi2.to_slicing_dict(pixel_size=pixel_size)
    assert slices2["y"] == slice(5, 6)
    assert slices2["x"] == slice(21, 22)


def test_write_well_points_skips_missing_field(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    mask0 = np.zeros((16, 16), dtype=np.uint16)
    mask0[3, 3] = 1

    _write_well_points(
        container=container,
        layout=layout,
        well_name="C09",
        channel_name="Spots",
        field_masks={1: mask0, 2: None},
        field_features={1: pd.DataFrame({"label": [1], "Spots_area": [1]}), 2: None},
    )

    table = container.get_generic_roi_table("Spots")
    assert len(table.rois()) == 1


def test_write_well_points_supports_3d_masks_with_a_z_slice(tmp_path):
    """A real z-stack's point-object channel is read as a 3D (Z, Y, X) mask
    (matching how blob channels are always read) -- each point's ROI must
    then carry a "z" slice at that point's own world z-coordinate, not just
    y/x."""
    container = _make_well_container(tmp_path, canvas_shape=(1, 3, 1, 16, 32))
    layout = _make_well_layout(tile_shape=(1, 1, 3, 16, 16), canvas_shape=(1, 1, 3, 16, 32))

    mask0 = np.zeros((3, 16, 16), dtype=np.uint16)
    mask0[2, 3, 3] = 1  # a single point on the third z-plane

    _write_well_points(
        container=container,
        layout=layout,
        well_name="C09",
        channel_name="Spots",
        field_masks={1: mask0, 2: None},
        field_features={1: None, 2: None},
    )

    table = container.get_generic_roi_table("Spots")
    rois = table.rois()
    assert len(rois) == 1
    pixel_size = PixelSize(x=layout.pixel_size_x, y=layout.pixel_size_y, z=layout.pixel_size_z)
    slices = rois[0].to_slicing_dict(pixel_size=pixel_size)
    assert slices["z"] == slice(2, 3)
    assert slices["y"] == slice(3, 4)
    assert slices["x"] == slice(3, 4)


# --------------------------------------------------------------------------- #
# features.py -- feature-table attachment with the same per-field
# global-ID offset used for label placement
# --------------------------------------------------------------------------- #


def test_offset_feature_table_ids_shifts_label_and_parent_label():
    df = pd.DataFrame({"label": [1, 2], "parent_label": [1, 1], "area": [10, 20]})
    out = _offset_feature_table_ids(df, field_id=3, well_name="C09", well_offset=0)
    offset = 3 * MAX_OBJECTS_PER_FIELD
    assert out["label"].tolist() == [1 + offset, 2 + offset]
    assert out["parent_label"].tolist() == [1 + offset, 1 + offset]
    assert out["area"].tolist() == [10, 20]


def test_offset_feature_table_ids_adds_human_readable_global_id():
    """The numeric label/parent_label offset has no way to carry the well
    name (it's an integer pixel value) -- global_id (text) is the
    traceability companion, built from the same (well, field_id, local_id),
    so it should never disagree with the numeric label."""
    df = pd.DataFrame({"label": [1, 2]})
    out = _offset_feature_table_ids(df, field_id=4, well_name="C09", well_offset=0)
    assert out["global_id"].tolist() == ["C09_0004_0000001", "C09_0004_0000002"]


def test_offset_feature_table_ids_adds_plate_wide_global_id_numeric():
    df = pd.DataFrame({"label": [1, 2]})
    well_offset = WELL_LABEL_OFFSET_STEP * 5
    out = _offset_feature_table_ids(df, field_id=1, well_name="C09", well_offset=well_offset)
    expected_label = 1 * MAX_OBJECTS_PER_FIELD
    assert out["global_id_numeric"].tolist() == [well_offset + expected_label + 1, well_offset + expected_label + 2]
    assert out["global_id_numeric"].dtype == np.int64


def test_offset_feature_table_ids_does_not_mutate_input():
    df = pd.DataFrame({"label": [1]})
    _offset_feature_table_ids(df, field_id=1, well_name="C09", well_offset=0)
    assert df["label"].tolist() == [1]


def test_write_well_features_matches_label_ids(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    field0[10:14, 10:14] = 2
    field1 = np.zeros((16, 16), dtype=np.uint16)
    field1[2:6, 2:6] = 1

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: field1})

    features0 = pd.DataFrame({"label": [1, 2], "Nuclei_area": [16, 16]})
    features1 = pd.DataFrame({"label": [1], "Nuclei_area": [16]})
    well_offset = well_label_offset(row_idx=2, col_idx=8)
    _write_well_features(
        container=container,
        label_name="Nuclei",
        field_dataframes={1: features0, 2: features1},
        well_name="C09",
        well_offset=well_offset,
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    label_ids = set(np.unique(container.get_label("Nuclei").get_as_numpy()).tolist()) - {0}
    assert set(df["label"].tolist()) == label_ids
    assert set(df["global_id"].tolist()) == {"C09_0001_0000001", "C09_0001_0000002", "C09_0002_0000001"}
    assert set(df["global_id_numeric"].tolist()) == {well_offset + label_id for label_id in label_ids}


def test_write_well_features_field_with_missing_measurements_contributes_no_rows(tmp_path):
    container = _make_well_container(tmp_path)
    layout = _make_well_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    field1 = np.zeros((16, 16), dtype=np.uint16)
    field1[2:6, 2:6] = 1

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: field1})

    features0 = pd.DataFrame({"label": [1], "Nuclei_area": [16]})
    # field 2's measurements are missing entirely (e.g. quantification failed)
    _write_well_features(
        container=container,
        label_name="Nuclei",
        field_dataframes={1: features0, 2: None},
        well_name="C09",
        well_offset=0,
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    assert len(df) == 1
    assert df["label"].iloc[0] == 1 * MAX_OBJECTS_PER_FIELD + 1
