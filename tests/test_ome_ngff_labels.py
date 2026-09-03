"""Tests for blimp.ome_ngff.labels -- segmentation label placement with
reproducible global IDs, and the point-object GenericRoiTable path."""
from pathlib import Path

import ngio
import numpy as np
import pandas as pd
import pytest

from blimp.ome_ngff.labels import (
    fov_object_id,
    _offset_label_ids,
    _write_well_labels,
    _write_well_points,
    MAX_OBJECTS_PER_FIELD,
)
from blimp.ome_ngff.layout import FieldLayout


def _make_field_layout(**overrides):
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


def _make_container(tmp_path: Path, canvas_shape=(1, 1, 1, 16, 32)):
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
    container = _make_container(tmp_path)
    layout = _make_field_layout()

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
    container = _make_container(tmp_path)
    layout = _make_field_layout()

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
    container = _make_container(tmp_path, canvas_shape=(1, 1, 3, 16, 32))
    layout = _make_field_layout(tile_shape=(1, 1, 3, 16, 16), canvas_shape=(1, 1, 3, 16, 32))

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
    container = _make_container(tmp_path)
    layout = _make_field_layout()

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
    container = _make_container(tmp_path)
    layout = _make_field_layout()

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
