"""Tests for blimp.ome_ngff.metadata -- NGFF 0.5 metadata construction and
pyramid downsampling shared across every OME-NGFF writer."""
import numpy as np

from blimp.ome_ngff.metadata import _downsample_yx, _build_ngff_v05_metadata


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
