"""Quantification tests built on synthetic images.

The existing ``test_quantify.py`` requires the downloaded reference dataset, so
none of it can run offline. These tests construct small ``AICSImage`` objects
with analytically known properties, which makes the expected feature values
exact rather than regression-recorded.
"""
import numpy as np
import pandas as pd
import pytest
from aicsimageio import AICSImage

from blimp.processing.quantify import (
    _quantify_single_timepoint_2D,
    _quantify_single_timepoint_3D,
    aggregate_and_merge_features,
    border_objects,
    border_objects_XY_3D,
    quantify,
)

# --------------------------------------------------------------------------
# Synthetic image builders
# --------------------------------------------------------------------------


def _image_2D(arrays, channel_names):
    """Stack 2D arrays (YX) into a TCZYX AICSImage with one T and one Z."""
    stack = np.stack(arrays)[np.newaxis, :, np.newaxis, :, :]
    return AICSImage(stack, channel_names=list(channel_names))


def _image_3D(arrays, channel_names, pixel_sizes=(1.0, 1.0, 1.0)):
    """Stack 3D arrays (ZYX) into a TCZYX AICSImage with one T.

    ``physical_pixel_sizes`` is required for 3D morphology features, so it is
    set to isotropic unit spacing to keep expected volumes in voxel units.
    """
    from aicsimageio.types import PhysicalPixelSizes

    stack = np.stack(arrays)[np.newaxis, ...]
    return AICSImage(
        stack,
        channel_names=list(channel_names),
        physical_pixel_sizes=PhysicalPixelSizes(*pixel_sizes),
    )


@pytest.fixture
def two_squares_2D():
    """Two labelled 4x4 squares with known areas and uniform intensities.

    Object 1: 16 px at intensity 100. Object 2: 16 px at intensity 300.
    Neither touches the image border.
    """
    label = np.zeros((32, 32), dtype=np.uint16)
    label[4:8, 4:8] = 1
    label[20:24, 20:24] = 2

    intensity = np.zeros((32, 32), dtype=np.uint16)
    intensity[4:8, 4:8] = 100
    intensity[20:24, 20:24] = 300

    return _image_2D([intensity], ["DAPI"]), _image_2D([label], ["nuclei"])


# --------------------------------------------------------------------------
# border_objects
# --------------------------------------------------------------------------


def test_border_objects_returns_one_row_per_label():
    label = np.zeros((10, 10), dtype=np.uint16)
    label[0:2, 0:2] = 1
    label[4:6, 4:6] = 2
    label[8:10, 8:10] = 3
    out = border_objects(label)
    assert isinstance(out, pd.DataFrame)
    assert sorted(out["label"]) == [1, 2, 3]


def test_border_objects_identifies_each_edge():
    """An object on any of the four edges must be flagged."""
    for sl in (np.s_[0:3, 4:7], np.s_[13:16, 4:7], np.s_[4:7, 0:3], np.s_[4:7, 13:16]):
        label = np.zeros((16, 16), dtype=np.uint16)
        label[sl] = 1
        label[8:10, 8:10] = 2  # interior control
        out = border_objects(label).set_index("label")
        assert bool(out.loc[1, "is_border"]) is True
        assert bool(out.loc[2, "is_border"]) is False


def test_border_objects_empty_label_image():
    out = border_objects(np.zeros((8, 8), dtype=np.uint16))
    assert len(out) == 0


def test_border_objects_XY_3D_ignores_z_edges():
    """An object touching only the top/bottom Z plane is not an XY border
    object -- z extent is expected to be clipped by the acquisition."""
    label = np.zeros((5, 16, 16), dtype=np.uint16)
    label[0, 6:9, 6:9] = 1  # first z plane, interior in XY
    label[2, 0:3, 6:9] = 2  # mid z, touches Y=0
    label_image = _image_3D([label], ["nuclei"])
    out = border_objects_XY_3D(label_image).set_index("label")
    assert bool(out.loc[1, "is_border_XY"]) is False
    assert bool(out.loc[2, "is_border_XY"]) is True


# --------------------------------------------------------------------------
# 2D quantification: exact, analytically-known values
# --------------------------------------------------------------------------


def test_quantify_2D_areas_are_exact(two_squares_2D):
    intensity, label = two_squares_2D
    df = _quantify_single_timepoint_2D(intensity, label, measure_object="nuclei")
    assert len(df) == 2
    areas = dict(zip(df["label"], df["nuclei_area"]))
    assert areas[1] == 16
    assert areas[2] == 16


def test_quantify_2D_mean_intensities_are_exact(two_squares_2D):
    intensity, label = two_squares_2D
    df = _quantify_single_timepoint_2D(
        intensity, label, measure_object="nuclei", intensity_channels="DAPI"
    )
    mean_col = [c for c in df.columns if "mean" in c.lower() and "DAPI" in c]
    assert mean_col, f"no DAPI mean column in {list(df.columns)}"
    means = dict(zip(df["label"], df[mean_col[0]]))
    assert means[1] == pytest.approx(100.0)
    assert means[2] == pytest.approx(300.0)


def test_quantify_2D_sum_intensity_is_area_times_value(two_squares_2D):
    intensity, label = two_squares_2D
    df = _quantify_single_timepoint_2D(
        intensity, label, measure_object="nuclei", intensity_channels="DAPI"
    )
    sum_col = [c for c in df.columns if "sum" in c.lower() and "DAPI" in c]
    assert sum_col
    sums = dict(zip(df["label"], df[sum_col[0]]))
    assert sums[1] == pytest.approx(16 * 100)
    assert sums[2] == pytest.approx(16 * 300)


def test_quantify_2D_sd_of_uniform_object_is_zero(two_squares_2D):
    intensity, label = two_squares_2D
    df = _quantify_single_timepoint_2D(
        intensity, label, measure_object="nuclei", intensity_channels="DAPI"
    )
    sd_col = [c for c in df.columns if "sd" in c.lower() and "DAPI" in c]
    assert sd_col
    assert np.allclose(df[sd_col[0]].to_numpy(), 0.0)


def test_quantify_2D_channel_by_index_matches_channel_by_name(two_squares_2D):
    intensity, label = two_squares_2D
    by_name = _quantify_single_timepoint_2D(
        intensity, label, measure_object="nuclei", intensity_channels="DAPI"
    )
    by_index = _quantify_single_timepoint_2D(
        intensity, label, measure_object="nuclei", intensity_channels=0
    )
    pd.testing.assert_frame_equal(by_name, by_index)


def test_quantify_2D_border_flag_present(two_squares_2D):
    intensity, label = two_squares_2D
    df = _quantify_single_timepoint_2D(intensity, label, measure_object="nuclei")
    border_cols = [c for c in df.columns if "border" in c.lower()]
    assert border_cols
    assert not df[border_cols[0]].any(), "no object touches the border in this fixture"


def test_quantify_2D_detects_border_object():
    label = np.zeros((32, 32), dtype=np.uint16)
    label[0:4, 4:8] = 1  # touches Y=0
    label[20:24, 20:24] = 2
    intensity = (label > 0).astype(np.uint16) * 100
    df = _quantify_single_timepoint_2D(
        _image_2D([intensity], ["DAPI"]), _image_2D([label], ["nuclei"]), measure_object="nuclei"
    )
    border_col = [c for c in df.columns if "border" in c.lower()][0]
    flags = dict(zip(df["label"], df[border_col]))
    assert bool(flags[1]) is True
    assert bool(flags[2]) is False


def test_quantify_2D_texture_features_are_added(two_squares_2D):
    intensity, label = two_squares_2D
    without = _quantify_single_timepoint_2D(intensity, label, measure_object="nuclei")
    with_tex = _quantify_single_timepoint_2D(
        intensity,
        label,
        measure_object="nuclei",
        calculate_textures=True,
        texture_channels="DAPI",
    )
    assert len(with_tex.columns) > len(without.columns)


def test_quantify_2D_empty_label_image_returns_empty_frame():
    label = np.zeros((16, 16), dtype=np.uint16)
    intensity = np.zeros((16, 16), dtype=np.uint16)
    df = _quantify_single_timepoint_2D(
        _image_2D([intensity], ["DAPI"]), _image_2D([label], ["nuclei"]), measure_object="nuclei"
    )
    assert len(df) == 0


def test_quantify_2D_mismatched_shapes_raise():
    """Regression: check_uniform_dimension_sizes() returns a bool that all three
    quantify call sites discarded, so mismatched images were quantified
    silently against whichever array was indexed first."""
    label = _image_2D([np.zeros((16, 16), dtype=np.uint16)], ["nuclei"])
    intensity = _image_2D([np.zeros((32, 32), dtype=np.uint16)], ["DAPI"])
    with pytest.raises(ValueError, match="matching dimension sizes"):
        _quantify_single_timepoint_2D(intensity, label, measure_object="nuclei")


# --------------------------------------------------------------------------
# 3D quantification
# --------------------------------------------------------------------------


@pytest.fixture
def two_cubes_3D():
    """Two 3x3x3 labelled cubes (27 voxels each) at intensities 50 and 150."""
    label = np.zeros((7, 24, 24), dtype=np.uint16)
    label[1:4, 4:7, 4:7] = 1
    label[3:6, 14:17, 14:17] = 2
    intensity = np.zeros((7, 24, 24), dtype=np.uint16)
    intensity[1:4, 4:7, 4:7] = 50
    intensity[3:6, 14:17, 14:17] = 150
    return _image_3D([intensity], ["DAPI"]), _image_3D([label], ["nuclei"])


def test_quantify_3D_voxel_counts_are_exact(two_cubes_3D):
    intensity, label = two_cubes_3D
    df = _quantify_single_timepoint_3D(intensity, label, measure_object="nuclei")
    assert len(df) == 2
    counts = dict(zip(df["label"], df["nuclei_3D_number_of_voxels"]))
    assert counts[1] == 27
    assert counts[2] == 27


def test_quantify_3D_physical_volume_scales_with_pixel_size(two_cubes_3D):
    """With 1 um isotropic spacing, 27 voxels = 27 um^3 = 0.027 pL."""
    intensity, label = two_cubes_3D
    df = _quantify_single_timepoint_3D(intensity, label, measure_object="nuclei")
    volumes = dict(zip(df["label"], df["nuclei_3D_physical_volume_pL"]))
    assert volumes[1] == pytest.approx(0.027)


def test_quantify_3D_MIP_area_is_projected_footprint(two_cubes_3D):
    """A 3x3x3 cube projects to a 3x3 = 9 px footprint."""
    intensity, label = two_cubes_3D
    df = _quantify_single_timepoint_3D(intensity, label, measure_object="nuclei")
    assert np.allclose(df["nuclei_3D_MIP_area"].to_numpy(), 9.0)


def test_quantify_3D_requires_physical_pixel_sizes():
    """3D morphology cannot be computed without voxel spacing, and must say so
    rather than silently reporting volumes in the wrong units."""
    lab = np.zeros((5, 16, 16), dtype=np.uint16)
    lab[1:4, 4:7, 4:7] = 1
    no_pps = AICSImage(lab[np.newaxis, ...], channel_names=["nuclei"])
    with pytest.raises(ValueError, match="physical_pixel_sizes"):
        _quantify_single_timepoint_3D(no_pps, no_pps, measure_object="nuclei")


def test_quantify_3D_mean_intensities_are_exact(two_cubes_3D):
    intensity, label = two_cubes_3D
    df = _quantify_single_timepoint_3D(
        intensity, label, measure_object="nuclei", intensity_channels="DAPI"
    )
    means = dict(zip(df["label"], df["nuclei_3D_intensity_mean_DAPI"]))
    assert means[1] == pytest.approx(50.0)
    assert means[2] == pytest.approx(150.0)


# --------------------------------------------------------------------------
# aggregate_and_merge_features
# --------------------------------------------------------------------------


def _parent_and_child():
    parent = pd.DataFrame({"label": [1, 2, 3], "nuclei_area": [100, 200, 300], "TimepointID": [0, 0, 0]})
    # parent 1 has two spots, parent 2 has one, parent 3 has none
    child = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "parent_label": [1, 1, 2],
            "spots_intensity": [10.0, 20.0, 40.0],
            "TimepointID": [0, 0, 0],
        }
    )
    return parent, child


def test_aggregate_counts_children_per_parent():
    parent, child = _parent_and_child()
    out = aggregate_and_merge_features([parent, child], parent_index=0, object_names=["nuclei", "spots"])
    counts = dict(zip(out["label"], out["spots_count"]))
    assert counts[1] == 2
    assert counts[2] == 1
    assert counts[3] == 0, "parents with no children must get count 0, not NaN"


def test_aggregate_count_column_is_integer_dtype():
    """Regression: the fillna(0) used a different column filter on each side of
    the assignment, which could leave float NaN counts behind."""
    parent, child = _parent_and_child()
    out = aggregate_and_merge_features([parent, child], parent_index=0, object_names=["nuclei", "spots"])
    assert pd.api.types.is_integer_dtype(out["spots_count"])
    assert out["spots_count"].notna().all()


def test_aggregate_sums_and_means_are_exact():
    parent, child = _parent_and_child()
    out = aggregate_and_merge_features(
        [parent, child], parent_index=0, object_names=["nuclei", "spots"]
    ).set_index("label")
    assert out.loc[1, "spots_intensity_sum"] == pytest.approx(30.0)
    assert out.loc[1, "spots_intensity_mean"] == pytest.approx(15.0)
    assert out.loc[1, "spots_intensity_min"] == pytest.approx(10.0)
    assert out.loc[1, "spots_intensity_max"] == pytest.approx(20.0)
    assert out.loc[2, "spots_intensity_sum"] == pytest.approx(40.0)


def test_aggregate_preserves_all_parent_rows():
    parent, child = _parent_and_child()
    out = aggregate_and_merge_features([parent, child], parent_index=0, object_names=["nuclei", "spots"])
    assert len(out) == len(parent)
    assert sorted(out["label"]) == [1, 2, 3]


def test_aggregate_does_not_aggregate_label_columns():
    parent, child = _parent_and_child()
    out = aggregate_and_merge_features([parent, child], parent_index=0, object_names=["nuclei", "spots"])
    for forbidden in ("label_sum", "label_mean", "parent_label_sum", "TimepointID_sum"):
        assert forbidden not in out.columns


def test_aggregate_rejects_mismatched_object_names():
    parent, child = _parent_and_child()
    with pytest.raises(ValueError, match="object names"):
        aggregate_and_merge_features([parent, child], parent_index=0, object_names=["nuclei"])


def test_aggregate_rejects_out_of_range_parent_index():
    parent, child = _parent_and_child()
    with pytest.raises(ValueError, match="parent_index"):
        aggregate_and_merge_features([parent, child], parent_index=5, object_names=["nuclei", "spots"])


# --------------------------------------------------------------------------
# quantify() end-to-end on synthetic data
# --------------------------------------------------------------------------


def _as_frame(result):
    """quantify() returns a DataFrame or a list of them depending on inputs."""
    if isinstance(result, list):
        assert len(result) == 1, f"expected a single feature table, got {len(result)}"
        return result[0]
    return result


def test_quantify_end_to_end_2D(two_squares_2D):
    intensity, label = two_squares_2D
    df = _as_frame(quantify(intensity, label, measure_objects="nuclei"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "TimepointID" in df.columns


def test_quantify_multiple_timepoints_are_all_present():
    label = np.zeros((32, 32), dtype=np.uint16)
    label[4:8, 4:8] = 1
    intensity = (label > 0).astype(np.uint16) * 100
    # two timepoints, identical content
    stack_l = np.stack([label, label])[:, np.newaxis, np.newaxis, :, :]
    stack_i = np.stack([intensity, intensity])[:, np.newaxis, np.newaxis, :, :]
    label_image = AICSImage(stack_l, channel_names=["nuclei"])
    intensity_image = AICSImage(stack_i, channel_names=["DAPI"])
    df = _as_frame(quantify(intensity_image, label_image, measure_objects="nuclei"))
    # blimp numbers timepoints from 1 (``timepoint + 1``), not from 0
    assert sorted(df["TimepointID"].unique()) == [1, 2]
    assert len(df) == 2
