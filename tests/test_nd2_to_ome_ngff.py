"""Tests for nd2_to_ome_ngff.py.

No fixture exercises the full convert_individual_nd2_to_ome_ngff pipeline
(well registration, tile placement, pyramid generation) end-to-end: the
existing reference .nd2 files are single-position with no well-plate
metadata, and there is no tool to produce a smaller real one -- the `nd2`
package (the most complete open reader for this proprietary format) has no
writer for the native container, only write_ome_zarr()/write_tiff(). The
pipeline was verified manually instead, against a real multi-position well
file: correct stitching (both "grid" and "continuous" placement -- see
get_field_layout), correct NGFF 0.5 metadata, and (for "continuous"
placement specifically) a 99.84% exact pixel match against the source data,
with differences confined to the expected tile-overlap zones from
last-write-wins placement. The tests below cover the pure logic (well-name
parsing, grid clustering, metadata construction, downsampling) that
pipeline is built from.
"""
from typing import Any, Dict
from pathlib import Path
import logging

from ngio.ome_zarr_meta.ngio_specs import PixelSize
import zarr
import numpy as np
import pytest

from blimp.constants import blimp_config
from blimp.preprocessing.nd2_to_ome_ngff import (
    FieldLayout,
    _WELL_NAME_RE,
    _downsample_yx,
    open_well_image,
    _parse_well_name,
    get_field_layout,
    _cluster_grid_index,
    _build_fov_roi_table,
    _build_ngff_v05_metadata,
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
    (confirmed against a real .nd2 file) -- clustering should collapse that
    jitter to exactly two clusters per axis, not four."""
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


def test_open_well_image_raises_clear_error_for_missing_kind(tmp_path):
    """No multi-position .nd2 fixture exists to exercise the full write
    pipeline end-to-end (see module docstring), so this builds just enough
    of a well's on-disk structure by hand -- a "mip" image directory plus
    the well-level ``ome.well.images`` metadata a real conversion would
    produce -- to exercise open_well_image's validation in isolation."""
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


@pytest.mark.data
def test_get_field_layout_raises_for_single_position_file_without_well_name(_ensure_test_data):
    """Real single-position .nd2 fixtures have no XYPosLoop and no 'WellXX'
    filename, so neither well-identity signal is available -- this should
    fail clearly rather than silently guessing a well."""
    testdata_config = blimp_config.get_data_config("testdata")
    nd2_path = Path(testdata_config.DATASET_DIR) / "illumination_correction" / "221103_brightfield_488_568_647_1.nd2"
    if not nd2_path.exists():
        pytest.skip(f"Reference file not found: {nd2_path}")

    with pytest.raises(ValueError, match="Could not determine a well identifier"):
        get_field_layout(nd2_path)
