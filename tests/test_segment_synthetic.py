from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.processing.segment import (
    compute_rescaling_limits,
    mask_child_objects_by_parent,
    resolve_multi_parent_objects,
    _resolve_single_measure_object,
)


def _label_image_2D(parent, child):
    """Stack two 2D label arrays (YX) into a TCZYX AICSImage."""
    stack = np.stack([parent, child])[np.newaxis, :, np.newaxis, :, :]
    return AICSImage(stack, channel_names=["Parent", "Child"])


def _label_image_3D(parent, child):
    """Stack two 3D label arrays (ZYX) into a TCZYX AICSImage."""
    stack = np.stack([parent, child])[np.newaxis, ...]
    return AICSImage(stack, channel_names=["Parent", "Child"])


EXPECTED_KEPT_COLUMNS = [3, 4, 5, 6, 7]


@pytest.fixture
def straddling_child_2D():
    """One child object straddling two parents, overlapping the first more.

    Parent 1 occupies columns 0-7, parent 2 columns 8-15. The child spans
    columns 3-9, so 5 of its 7 columns lie in parent 1 and 2 in parent 2.
    Resolution should keep only the parent-1 portion. The overlap is
    deliberately lopsided (5 vs 2) so the assertion tests the "largest
    overlap" rule rather than tie-breaking behaviour.
    """
    parent = np.zeros((16, 16), dtype=np.uint16)
    parent[:, 0:8] = 1
    parent[:, 8:16] = 2

    child = np.zeros((16, 16), dtype=np.uint16)
    child[4:8, 3:10] = 1
    return _label_image_2D(parent, child)


EXPECTED_KEPT_COLUMNS_SECOND_PARENT = [8, 9, 10, 11, 12]


@pytest.fixture
def straddling_child_2D_second_parent_wins():
    """As above, but the larger overlap is with parent 2 rather than parent 1.

    The child spans columns 6-12: 2 columns in parent 1 and 5 in parent 2. The
    winning parent is therefore neither the first label nor the first column,
    so an implementation that picks the lowest-numbered overlapping parent is
    distinguishable from one that picks the largest overlap.
    """
    parent = np.zeros((16, 16), dtype=np.uint16)
    parent[:, 0:8] = 1
    parent[:, 8:16] = 2

    child = np.zeros((16, 16), dtype=np.uint16)
    child[4:8, 6:13] = 1
    return _label_image_2D(parent, child)


def _child_columns(image, timepoint=0):
    """Columns in which the child channel has any labelled pixel."""
    child = image.data[timepoint, 1, 0, :, :]
    return sorted(np.unique(np.nonzero(child)[1]).tolist())


def test_resolve_multi_parent_not_in_place_returns_new_image(straddling_child_2D):
    """in_place=False returns a new image and leaves the input untouched."""
    before = straddling_child_2D.data.copy()

    result = resolve_multi_parent_objects(
        straddling_child_2D, measure_object="Child", parent_object="Parent", in_place=False
    )

    assert result is not None
    # the input is unmodified
    np.testing.assert_array_equal(straddling_child_2D.data, before)
    # the child is trimmed to the parent it overlaps most
    assert _child_columns(result) == EXPECTED_KEPT_COLUMNS


def test_resolve_multi_parent_in_place_modifies_input(straddling_child_2D):
    """in_place=True writes through to the input and returns None."""
    result = resolve_multi_parent_objects(
        straddling_child_2D, measure_object="Child", parent_object="Parent", in_place=True
    )

    assert result is None
    assert _child_columns(straddling_child_2D) == EXPECTED_KEPT_COLUMNS


def test_resolve_multi_parent_keeps_largest_overlap_not_lowest_label(
    straddling_child_2D_second_parent_wins,
):
    """The child is assigned by overlap area, not by parent label order."""
    result = resolve_multi_parent_objects(
        straddling_child_2D_second_parent_wins,
        measure_object="Child",
        parent_object="Parent",
        in_place=False,
    )

    assert result is not None
    assert _child_columns(result) == EXPECTED_KEPT_COLUMNS_SECOND_PARENT


def test_resolve_multi_parent_3D(straddling_child_2D):
    """The 3D branch takes a different indexed assignment than the 2D branch."""
    parent = np.zeros((4, 16, 16), dtype=np.uint16)
    parent[:, :, 0:8] = 1
    parent[:, :, 8:16] = 2

    child = np.zeros((4, 16, 16), dtype=np.uint16)
    child[1:3, 4:8, 3:10] = 1

    image = _label_image_3D(parent, child)
    result = resolve_multi_parent_objects(image, measure_object="Child", parent_object="Parent", in_place=False)

    assert result is not None
    child_out = result.data[0, 1, :, :, :]
    assert sorted(np.unique(np.nonzero(child_out)[2]).tolist()) == EXPECTED_KEPT_COLUMNS


def test_resolve_single_measure_object_requires_stack_when_not_in_place(straddling_child_2D):
    """in_place=False without a destination array must raise ValueError."""
    with pytest.raises(ValueError, match="new_label_stack must be provided"):
        _resolve_single_measure_object(
            straddling_child_2D,
            measure_object_index=1,
            parent_object_index=0,
            timepoint=0,
            in_place=False,
            new_label_stack=None,
        )


def _intensity_image_2D(array):
    """Wrap a single 2D intensity array (YX) into a TCZYX AICSImage."""
    stack = array[np.newaxis, np.newaxis, np.newaxis, :, :]
    return AICSImage(stack, channel_names=["Nuclei"])


# 101 evenly spaced integers 0..100: with n=101 points, np.percentile(_, p) == p
# exactly for integer p, so percentile-based assertions below can use plain equality.
_LINEAR_101 = np.arange(101, dtype=np.uint16)


def test_compute_rescaling_limits_single_array():
    """A bare (non-list) array input returns the expected percentile pair."""
    assert compute_rescaling_limits(_LINEAR_101, percentile=(1.0, 99.0)) == (1.0, 99.0)


def test_compute_rescaling_limits_single_element_list_matches_bare_input():
    """Wrapping the same image in a 1-element list gives an identical result."""
    assert compute_rescaling_limits([_LINEAR_101]) == compute_rescaling_limits(_LINEAR_101)


def test_compute_rescaling_limits_aggregation_mean_vs_median():
    """mean and median give different, individually correct results when images differ.

    Three images with percentile pairs (1, 99), (101, 199), (1001, 1099): the mean is
    pulled toward the high outlier while the median lands exactly on the middle image,
    so a wrong aggregation choice is caught rather than masked by coincidentally equal
    results.
    """
    low_image = _LINEAR_101
    mid_image = _LINEAR_101 + 100
    high_image = _LINEAR_101 + 1000
    images = [low_image, mid_image, high_image]

    mean_result = compute_rescaling_limits(images, aggregation="mean")
    median_result = compute_rescaling_limits(images, aggregation="median")

    assert mean_result == pytest.approx(((1 + 101 + 1001) / 3, (99 + 199 + 1099) / 3))
    assert median_result == (101.0, 199.0)
    assert mean_result != median_result


def test_compute_rescaling_limits_input_types_agree(tmp_path):
    """ndarray, AICSImage, and on-disk path inputs all give the same result for the same data.

    Tiled to a (101, 3) shape (no singleton spatial dimension) since round-tripping a
    Y=1 or X=1 image through an OME-TIFF write/read hits an unrelated aicsimageio
    reshape quirk with singleton spatial dimensions.
    """
    array = np.tile(_LINEAR_101.reshape(101, 1), (1, 3))
    array_result = compute_rescaling_limits(array)

    image = _intensity_image_2D(array)
    image_result = compute_rescaling_limits(image, channel=0)

    image_path = tmp_path / "intensity.ome.tiff"
    image.save(image_path)
    path_result = compute_rescaling_limits(image_path, channel=0)

    assert image_result == array_result
    assert path_result == array_result


def test_compute_rescaling_limits_empty_images_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_rescaling_limits([])


@pytest.mark.parametrize("percentile", [(99.0, 1.0), (-5.0, 99.0), (1.0, 101.0), (50.0, 50.0)])
def test_compute_rescaling_limits_invalid_percentile_raises(percentile):
    with pytest.raises(ValueError, match="percentile"):
        compute_rescaling_limits(_LINEAR_101, percentile=percentile)


def test_compute_rescaling_limits_invalid_aggregation_raises():
    with pytest.raises(ValueError, match="aggregation"):
        compute_rescaling_limits(_LINEAR_101, aggregation="max")


def test_mask_child_objects_by_parent_not_in_place():
    """Child pixels outside any parent are removed; the input is untouched."""
    parent = np.zeros((16, 16), dtype=np.uint16)
    parent[4:8, 4:8] = 1

    child = np.zeros((16, 16), dtype=np.uint16)
    child[4:8, 4:8] = 1  # inside the parent
    child[12:14, 12:14] = 2  # outside any parent

    image = _label_image_2D(parent, child)
    before = image.data.copy()

    result = mask_child_objects_by_parent(image, parent_object="Parent", in_place=False)

    assert result is not None
    np.testing.assert_array_equal(image.data, before)
    remaining = np.unique(result.data[0, 1, 0, :, :])
    assert remaining.tolist() == [0, 1]
