"""Regression tests pinning previously-fixed defects.

Each test here corresponds to a specific bug found during the code audit. They
are deliberately written against small in-memory arrays so the whole module runs
offline in under a second. If one of these fails, a fixed bug has returned.
"""
from pathlib import Path
import pickle
import logging
import warnings

from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.data import get_filename_from_content_disposition
from blimp.utils import safe_log10
from blimp.constants import blimp_config
from blimp.processing.quantify import border_objects
from blimp.preprocessing.registration import (
    register_2D,
    transform_2D,
    TransformationParameters,
)
from blimp.preprocessing.illumination_correction import (
    pixel_z_score,
    _floor_zero_std,
    _ZERO_STD_WARNED,
)
import blimp.utils as utils

# utils.estimate_focus_plane: the type guard validated ``crop`` twice, so the
# second check (`isinstance(crop, AICSImage)`) was always False and the function
# raised TypeError for every valid call.


def _blobs(size: int = 96, n_blobs: int = 12, seed: int = 0) -> np.ndarray:
    """A synthetic field of Gaussian 'nuclei' -- structured, not pure noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:size, :size]
    img = np.zeros((size, size))
    for _ in range(n_blobs):
        y, x = rng.integers(10, size - 10, 2)
        img += 3000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 6.0**2))
    return img + rng.normal(0, 50, (size, size))


def _zstack_with_focus_at(focus_z: int, n_z: int = 7, size: int = 96) -> AICSImage:
    """Build a TCZYX uint16 stack defocused by a Gaussian away from ``focus_z``."""
    gaussian_filter = pytest.importorskip("scipy.ndimage").gaussian_filter
    base = _blobs(size=size)
    planes = []
    for z in range(n_z):
        sigma = 1.5 * abs(z - focus_z)
        plane = gaussian_filter(base, sigma) if sigma > 0 else base
        planes.append(np.clip(plane, 0, 65535).astype(np.uint16))
    arr = np.stack(planes)[np.newaxis, np.newaxis, ...]
    return AICSImage(arr)


def test_estimate_focus_plane_accepts_valid_image():
    """Valid input must not raise (regression: wrong-variable type guard)."""
    image = _zstack_with_focus_at(3)
    plane = utils.estimate_focus_plane(image, C=0)
    assert isinstance(plane, int)
    assert 0 <= plane < image.dims.Z


@pytest.mark.parametrize("focus_z", [1, 3, 5])
def test_estimate_focus_plane_finds_sharpest_plane(focus_z):
    image = _zstack_with_focus_at(focus_z)
    assert utils.estimate_focus_plane(image, C=0) == focus_z


# utils._vollath_f4 accumulated in the input dtype, so uint16 products (up to
# ~4.3e9) wrapped modulo 65536. The metric was ~100% wrong on 16-bit camera
# data and could select the wrong focus plane.


def test_vollath_f4_is_dtype_independent():
    """The metric must not change when the same values are held as uint16."""
    base = np.clip(_blobs(), 0, 65535)
    as_uint16 = base.astype(np.uint16)
    from_int = utils._vollath_f4(as_uint16)
    from_float = utils._vollath_f4(as_uint16.astype(np.float64))
    assert from_float > 1e7, "test image should produce a large F4 value"
    assert from_int == pytest.approx(from_float, rel=1e-9)


def test_vollath_f4_does_not_overflow_on_saturated_uint16():
    """A saturated uint16 patch has a known exact F4 value."""
    arr = np.full((3, 3), 60000, dtype=np.uint16)
    # sum1 = 2 rows * 3 cols * 60000^2 ; sum2 = 1 row * 3 cols * 60000^2
    expected = (2 * 3 - 1 * 3) * 60000.0**2
    assert utils._vollath_f4(arr) == pytest.approx(expected)


def test_vollath_f4_decreases_with_blur():
    """Sharper images must score higher -- the property autofocus relies on."""
    gaussian_filter = pytest.importorskip("scipy.ndimage").gaussian_filter
    base = _blobs()
    scores = [
        utils._vollath_f4(np.clip(gaussian_filter(base, s) if s else base, 0, 65535).astype(np.uint16))
        for s in (0, 1, 2, 4, 8)
    ]
    assert scores == sorted(scores, reverse=True), f"F4 not monotonic in blur: {scores}"


def test_estimate_focus_plane_rejects_non_image():
    with pytest.raises(TypeError, match="AICSImage"):
        utils.estimate_focus_plane(np.zeros((4, 4)), C=0)


def test_estimate_focus_plane_rejects_bad_crop_type():
    image = _zstack_with_focus_at(2)
    with pytest.raises(TypeError, match="crop"):
        utils.estimate_focus_plane(image, crop=1, C=0)


def test_estimate_focus_plane_without_sliding_window():
    """max_pos was only bound inside the `sliding_window is not None` branch."""
    image = _zstack_with_focus_at(2)
    assert isinstance(utils.estimate_focus_plane(image, C=0, sliding_window=None), int)


# utils.convert_array_dtype: the dask branch called np.ndarray(arr) (the
# constructor) instead of converting, raising TypeError for dask input.


def test_convert_array_dtype_accepts_dask():
    da = pytest.importorskip("dask.array")
    arr = da.from_array(np.arange(12, dtype=np.uint8).reshape(3, 4), chunks=2)
    out = utils.convert_array_dtype(arr, np.uint16)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, np.arange(12).reshape(3, 4))


def test_convert_array_dtype_rejects_bad_dtype():
    with pytest.raises(TypeError, match="not recognised|not allowed"):
        utils.convert_array_dtype(np.zeros((2, 2)), np.complex128)


# illumination_correction.pixel_z_score
#  (a) zeros were set to 1e-10, log10'd to -10, then a second `== 0` test did
#      nothing -- the intended zero-masking never happened.
#  (b) the integer-rounding branch tested `original.dtype` after `original` had
#      been rebound to float64, so it was dead and results were truncated
#      instead of rounded.


def test_pixel_z_score_preserves_integer_dtype():
    original = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    mean_image = np.full((2, 2), 2.0)
    std_image = np.full((2, 2), 0.5)
    out = pixel_z_score(original, mean_image, std_image, 2.0, 0.5, log_transform=True)
    assert out.dtype == np.uint16


def test_pixel_z_score_rounds_rather_than_truncates():
    """With identity statistics the transform is a no-op; rounding must recover
    the input exactly rather than truncating downward."""
    original = np.array([[10, 11], [12, 13]], dtype=np.uint16)
    ones = np.ones((2, 2))
    out = pixel_z_score(original, ones, ones, 1.0, 1.0, log_transform=True)
    np.testing.assert_array_equal(out, original)


def test_pixel_z_score_handles_zero_pixels():
    """Zero pixels must not produce inf/nan after the log transform."""
    original = np.array([[0, 0], [5, 10]], dtype=np.uint16)
    ones = np.ones((2, 2))
    out = pixel_z_score(original, ones, ones, 1.0, 1.0, log_transform=True)
    assert np.all(np.isfinite(out.astype(np.float64)))


def test_pixel_z_score_float_input_unchanged_dtype():
    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ones = np.ones((2, 2))
    out = pixel_z_score(original, ones, ones, 1.0, 1.0, log_transform=False)
    assert out.dtype == np.float32


# illumination_correction: zero reference standard deviation
#  A per-pixel std of exactly 0 made the z-score division produce +/-inf (or
#  NaN for 0/0), which the cast to uint16 turned into 0 or 65535. The output
#  was a plausible-looking image with dead and saturated pixels, so nothing
#  failed and nothing warned. On the packaged two-image reference dataset this
#  hit 3.95% of pixels, at ordinary mid-range intensities.


def test_pixel_z_score_zero_std_does_not_produce_nonfinite():
    """A zero std must not leak inf/NaN into the corrected image."""
    original = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    mean_image = np.full((2, 2), 2.0)
    std_image = np.array([[0.0, 0.5], [0.25, 0.5]])
    out = pixel_z_score(original, mean_image, std_image, 2.0, 0.5, log_transform=True)
    assert np.all(np.isfinite(out.astype(np.float64)))


def test_pixel_z_score_zero_std_does_not_saturate_or_zero_the_pixel():
    """The failure mode was silent: affected pixels became 0 or the dtype max
    while looking like ordinary data. Guard the symptom directly."""
    original = np.array([[500, 500], [500, 500]], dtype=np.uint16)
    mean_image = np.full((2, 2), np.log10(500.0))
    std_image = np.array([[0.0, 0.1], [0.1, 0.1]])
    out = pixel_z_score(original, mean_image, std_image, np.log10(500.0), 0.1, log_transform=True)
    assert out[0, 0] not in (0, np.iinfo(np.uint16).max)


def test_floor_zero_std_uses_median_of_positive_values():
    std_image = np.array([[0.0, 1.0], [2.0, 3.0]])
    floored = _floor_zero_std(std_image)
    # median of the positive entries (1, 2, 3) is 2
    assert floored[0, 0] == 2.0


def test_floor_zero_std_leaves_positive_values_untouched():
    std_image = np.array([[0.0, 1.0], [2.0, 3.0]])
    floored = _floor_zero_std(std_image)
    np.testing.assert_array_equal(floored[std_image > 0], std_image[std_image > 0])


def test_floor_zero_std_does_not_mutate_input():
    """The caller's correction object must not be modified in place."""
    std_image = np.array([[0.0, 1.0], [2.0, 3.0]])
    before = std_image.copy()
    _floor_zero_std(std_image)
    np.testing.assert_array_equal(std_image, before)


def test_floor_zero_std_returns_input_when_no_zeros():
    std_image = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_floor_zero_std(std_image), std_image)


def test_floor_zero_std_all_zero_is_left_alone():
    """With no positive value to floor with there is nothing defensible to do;
    the caller is warned rather than handed an invented std."""
    std_image = np.zeros((2, 2))
    np.testing.assert_array_equal(_floor_zero_std(std_image), std_image)


def test_floor_zero_std_warns_with_affected_fraction(caplog):
    """The substitution is an assumption, so it must be visible in the log."""
    std_image = np.array([[0.0, 1.0], [2.0, 3.0]])
    # The warning is deduplicated per distinct (n_zero, size, floor) signature so
    # that a timelapse does not emit thousands of copies. Clear the cache, or this
    # assertion depends on whether an earlier test already logged this signature.
    _ZERO_STD_WARNED.clear()
    with caplog.at_level(logging.WARNING):
        _floor_zero_std(std_image)
    assert "25.00%" in caplog.text


# utils.safe_log10 -- zero and negative handling before the log transform
#  Zeros previously took two different paths that disagreed by ten decades:
#  `mean_std_welford` mapped a zero to log-space 0.0 (raw value 1) while
#  `pixel_z_score` clamped it to 1e-10 (log-space -10), giving z-scores of
#  order -3000 for a pixel whose neighbours sat near 3. Negative values were
#  worse: log10 of a negative is NaN, NaN is not caught by an `isinf` check,
#  so it entered the Welford accumulator and poisoned that pixel for every
#  subsequent reference image.


def test_safe_log10_maps_zero_to_zero():
    """log10(1) == 0, the convention the reference statistics already used."""
    array = np.array([[0, 1, 10], [100, 0, 1000]], dtype=np.uint16)
    out = safe_log10(array)
    assert out[0, 0] == 0.0
    assert out[1, 1] == 0.0


def test_safe_log10_matches_old_welford_behaviour():
    """The reference path previously log10'd then patched -inf back to 0. The
    new form must be numerically identical, so saved correction objects stay
    valid -- only the spurious log10(0) warning goes away."""
    array = np.array([[0, 1, 100], [200, 0, 300]], dtype=np.uint16)
    with np.errstate(divide="ignore"):
        old = np.log10(array.astype(np.float64))
    old[array == 0] = 0.0
    np.testing.assert_array_equal(safe_log10(array), old)


def test_safe_log10_produces_no_nonfinite_values():
    array = np.array([[0, 0], [0, 65535]], dtype=np.uint16)
    assert np.all(np.isfinite(safe_log10(array)))


def test_safe_log10_emits_no_numpy_warning_for_zeros():
    """The old form called log10(0) and warned before patching the result."""
    array = np.array([[0, 100]], dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        safe_log10(array)


def test_safe_log10_does_not_mutate_input():
    array = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    before = array.copy()
    safe_log10(array)
    np.testing.assert_array_equal(array, before)


def test_safe_log10_rejects_float_input():
    """Replacing zeros with 1 assumes 1 is one detector count. On floats
    normalised to [0, 1] that is the maximum of the data, not a floor."""
    with pytest.raises(TypeError, match="integer intensity array"):
        safe_log10(np.array([[0.0, 0.5]], dtype=np.float64))


def test_safe_log10_rejects_negative_values():
    """log10 of a negative is NaN, which `isinf` does not catch."""
    with pytest.raises(ValueError, match="non-negative"):
        safe_log10(np.array([[-500, 100]], dtype=np.int16))


def test_safe_log10_accepts_signed_integers_without_negatives():
    array = np.array([[0, 100]], dtype=np.int32)
    np.testing.assert_allclose(safe_log10(array), [[0.0, 2.0]])


def test_pixel_z_score_zero_pixels_agree_with_reference_convention():
    """A zero pixel must be treated as log-space 0 by `pixel_z_score`, the same
    value `mean_std_welford` used when building the statistics. The old 1e-10
    clamp mapped it to -10 instead."""
    original = np.array([[0, 100]], dtype=np.uint16)
    mean_image = np.zeros((1, 2))
    std_image = np.ones((1, 2))
    out = pixel_z_score(original, mean_image, std_image, 0.0, 1.0, log_transform=True)
    # z = (0 - 0) / 1 = 0  ->  corrected = 10**0 = 1
    assert out[0, 0] == 1


def test_floor_zero_std_warns_only_once_per_signature(caplog):
    """`pixel_z_score` runs per (T, C, Z) plane, so an undeduplicated warning
    would flood the log on a timelapse."""
    std_image = np.array([[0.0, 1.0], [2.0, 3.0]])
    _ZERO_STD_WARNED.clear()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            _floor_zero_std(std_image)
    assert caplog.text.count("Reference standard deviation is zero") == 1


# registration.TransformationParameters
#  (a) `self.from_resources` was assigned `str(from_file)` -- a copy-paste bug
#      that reported the wrong provenance.
#  (b) `if all([...]) is None` can never be True, so calling with no arguments
#      fell through to an unrelated error instead of the intended message.


def test_transformation_parameters_records_provenance_separately():
    params = TransformationParameters(transformation_mode="rigid")
    assert params.from_file is None
    assert params.from_resources is None


def test_transformation_parameters_requires_an_argument():
    with pytest.raises(ValueError, match="requires one of"):
        TransformationParameters()


def test_transformation_parameters_unknown_mode_is_reported():
    with pytest.raises((ValueError, KeyError)):
        TransformationParameters(transformation_mode="not-a-real-mode")


# registration.register_2D / transform_2D: itk elastix filters are only wrapped
# for itk.Image inputs; passing bare numpy arrays raised TemplateTypeError.


def test_register_2D_parameters_only_returns_parameters():
    rng = np.random.default_rng(1)
    fixed = rng.random((16, 16))
    moving = rng.random((16, 16))
    settings = TransformationParameters(transformation_mode="translation")
    params = register_2D(fixed, moving, settings, parameters_only=True)
    assert isinstance(params, TransformationParameters)


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.float32])
def test_register_2D_preserves_dtype(dtype):
    rng = np.random.default_rng(2)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        fixed = rng.integers(info.min, info.max, size=(16, 16)).astype(dtype)
        moving = rng.integers(info.min, info.max, size=(16, 16)).astype(dtype)
    else:
        fixed = rng.random((16, 16)).astype(dtype)
        moving = rng.random((16, 16)).astype(dtype)
    registered, params = register_2D(fixed, moving, TransformationParameters(transformation_mode="translation"))
    assert registered.dtype == dtype
    assert isinstance(params, TransformationParameters)


def test_register_2D_recovers_known_translation():
    """A pure shift must be recovered to sub-pixel accuracy."""
    rng = np.random.default_rng(3)
    base = rng.random((64, 64))
    shift_y, shift_x = 3, -2
    moving = np.roll(np.roll(base, shift_y, axis=0), shift_x, axis=1)
    settings = TransformationParameters(transformation_mode="translation")
    registered, _ = register_2D(base, moving, settings)
    # registered should resemble the fixed image more than the unaligned moving one
    err_before = np.mean(np.abs(base[8:-8, 8:-8] - moving[8:-8, 8:-8]))
    err_after = np.mean(np.abs(base[8:-8, 8:-8] - registered[8:-8, 8:-8]))
    assert err_after < err_before


def test_register_2D_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="[Ss]hape"):
        register_2D(
            np.zeros((8, 8)),
            np.zeros((8, 9)),
            TransformationParameters(transformation_mode="translation"),
        )


def test_register_2D_rejects_dtype_mismatch():
    with pytest.raises(TypeError, match="dtype"):
        register_2D(
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.float32),
            TransformationParameters(transformation_mode="translation"),
        )


def test_transform_2D_roundtrip():
    rng = np.random.default_rng(4)
    fixed = rng.random((32, 32))
    moving = np.roll(fixed, 2, axis=0)
    settings = TransformationParameters(transformation_mode="translation")
    registered, params = register_2D(fixed, moving, settings)
    transformed = transform_2D(moving, params)
    np.testing.assert_allclose(transformed, registered, atol=1e-5)


# quantify: the `_count` fill used `endswith("count")` on the right-hand side
# but `endswith("_count")` on the left, so the two column sets could differ.


def test_border_objects_flags_edge_touching_labels():
    label = np.zeros((16, 16), dtype=np.int32)
    label[0:3, 0:3] = 1  # touches border
    label[7:10, 7:10] = 2  # interior
    out = border_objects(label)
    flags = dict(zip(out["label"], out["is_border"]))
    assert flags[1]
    assert not flags[2]


# constants: the setters called os.path.abspath() unconditionally, so assigning
# None (which the getters explicitly warn about) raised TypeError.


def test_config_dirs_accept_none(monkeypatch):
    monkeypatch.setattr(blimp_config, "EXPERIMENT_DIR", None, raising=False)
    monkeypatch.setattr(blimp_config, "BASE_DATA_DIR", None, raising=False)
    assert blimp_config.EXPERIMENT_DIR is None
    assert blimp_config.BASE_DATA_DIR is None


def test_config_dirs_are_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr(blimp_config, "BASE_DATA_DIR", str(tmp_path), raising=False)
    assert Path(blimp_config.BASE_DATA_DIR).is_absolute()


# data: content-disposition parsing returned None and the caller then called
# Path(None). Also covers the rename to a PEP8 name.


@pytest.mark.parametrize(
    "header,expected",
    [
        ('attachment; filename="data.zip"', "data.zip"),
        ("attachment; filename=data.zip", "data.zip"),
        (None, None),
        ("attachment", None),
    ],
)
def test_get_filename_from_content_disposition(header, expected):
    assert get_filename_from_content_disposition(header) == expected


# image.BLImage: mutable default argument, Path(str) instead of Path(path), and
# pickle.load called on a path rather than a file handle.


def test_blimage_fs_kwargs_not_shared_between_instances():
    from blimp.image import BLImage

    arr = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
    a = BLImage(arr)
    b = BLImage(arr)
    # mutating one instance's kwargs must not leak into the class default
    assert a is not b
    import inspect

    default = inspect.signature(BLImage.__init__).parameters["fs_kwargs"].default
    assert default is None, "fs_kwargs must not be a shared mutable default"


def test_illumination_correction_objects_load_from_handle(tmp_path):
    """pickle.load must be given an open handle, not a Path."""
    from blimp.image import BLImage

    arr = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    img = BLImage(arr)
    payload = ["channel-0-correction", "channel-1-correction"]
    pkl = tmp_path / "illum.pkl"
    with open(pkl, "wb") as handle:
        pickle.dump(payload, handle)

    img._illumination_correction_file = pkl
    img._load_illumination_correction_objects()
    assert img.illumination_correction_objects == payload
