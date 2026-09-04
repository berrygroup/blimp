"""Tests for the shared blimp.ome_ngff writer core (layout, metadata,
plate/well registration, label/point placement, feature tables), used by
every source-format writer (nd2, TIFF-pipeline)."""
from typing import Any, Dict
from pathlib import Path
import logging

from ngio.ome_zarr_meta.ngio_specs import PixelSize
import ngio
import zarr
import numpy as np
import pandas as pd
import pytest

from blimp.ome_ngff.plate import open_well_image, ensure_plate_exists
from blimp.ome_ngff.labels import (
    fov_object_id,
    _offset_label_ids,
    _write_well_labels,
    _write_well_points,
    MAX_OBJECTS_PER_FIELD,
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


def test_fov_object_id_format():
    assert fov_object_id("C09", 4, 123) == "C09_0004_0000123"


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

    roi_table = container.get_masking_roi_table("Nuclei_ROI_table")
    assert len(roi_table.rois()) == 3


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
    """segment_nuclei_cellpose itself only ever produces 2D (Z==1) labels,
    but quantify() and the rest of segment.py explicitly support 3D label
    images too (e.g. from running cellpose's own do_3D=True mode directly).
    A genuinely 3D local array must be placed one real slice per Z-plane --
    if the placement logic ever regressed to assuming a flat 2D array, a
    single 2D mask would get broadcast (replicated) across every Z-plane
    instead, which this test would catch."""
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
    out = _offset_feature_table_ids(df, field_id=3, well_name="C09")
    offset = 3 * MAX_OBJECTS_PER_FIELD
    assert out["label"].tolist() == [1 + offset, 2 + offset]
    assert out["parent_label"].tolist() == [1 + offset, 1 + offset]
    assert out["area"].tolist() == [10, 20]


def test_offset_feature_table_ids_adds_human_readable_fov_object_id():
    """The numeric label/parent_label offset has no way to carry the well
    name (it's an integer pixel value) -- fov_object_id is the traceability
    companion, built from the same (well, field_id, local_id), so it should
    never disagree with the numeric global_id."""
    df = pd.DataFrame({"label": [1, 2]})
    out = _offset_feature_table_ids(df, field_id=4, well_name="C09")
    assert out["fov_object_id"].tolist() == ["C09_0004_0000001", "C09_0004_0000002"]


def test_offset_feature_table_ids_does_not_mutate_input():
    df = pd.DataFrame({"label": [1]})
    _offset_feature_table_ids(df, field_id=1, well_name="C09")
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
    _write_well_features(
        container=container,
        label_name="Nuclei",
        field_dataframes={1: features0, 2: features1},
        well_name="C09",
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    label_ids = set(np.unique(container.get_label("Nuclei").get_as_numpy()).tolist()) - {0}
    assert set(df["label"].tolist()) == label_ids
    assert set(df["fov_object_id"].tolist()) == {"C09_0001_0000001", "C09_0001_0000002", "C09_0002_0000001"}


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
        container=container, label_name="Nuclei", field_dataframes={1: features0, 2: None}, well_name="C09"
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    assert len(df) == 1
    assert df["label"].iloc[0] == 1 * MAX_OBJECTS_PER_FIELD + 1
