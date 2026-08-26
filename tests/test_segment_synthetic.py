"""Tests for the parent/child label-resolution helpers in ``blimp.processing.segment``.

``resolve_multi_parent_objects`` had no test coverage at all, and the coverage
report showed its per-channel helper ``_resolve_single_measure_object`` as
never entered. Both it and ``mask_child_objects_by_parent`` write their result
through an indexed assignment whose target depends on ``in_place``, so the two
branches are exercised separately here.

These build small ``AICSImage`` objects in memory, so they need neither the
downloaded reference dataset nor network access.
"""
from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.processing.segment import (
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


#: Columns of the child object that survive resolution in the fixtures below:
#: the child spans columns 3-9, of which 3-7 (5 columns) lie in parent 1 and
#: 8-9 (2 columns) in parent 2, so parent 1 wins on overlap.
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
    """The helper's in_place/new_label_stack invariant is enforced explicitly.

    Passing in_place=False without a destination array used to reach an indexed
    assignment on None and fail with an opaque
    ``TypeError: 'NoneType' object does not support item assignment``.
    """
    with pytest.raises(ValueError, match="new_label_stack must be provided"):
        _resolve_single_measure_object(
            straddling_child_2D,
            measure_object_index=1,
            parent_object_index=0,
            timepoint=0,
            in_place=False,
            new_label_stack=None,
        )


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
