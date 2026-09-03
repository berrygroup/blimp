"""Tests for blimp.ome_ngff.layout -- field-of-view layout and grid-clustering
math shared across every OME-NGFF writer."""
from typing import Any, Dict
import logging

from ngio.ome_zarr_meta.ngio_specs import PixelSize
import numpy as np
import pytest

from blimp.ome_ngff.layout import (
    FieldLayout,
    _WELL_NAME_RE,
    _parse_well_name,
    _cluster_grid_index,
    _exact_pixel_offset,
    _build_fov_roi_table,
)


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
