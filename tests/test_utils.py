from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
import numpy as np
import pytest
import dask.array as da

import blimp.utils


def test_axis_str_to_int():
    assert blimp.utils._axis_str_to_int("T") == 0
    assert blimp.utils._axis_str_to_int("C") == 1
    assert blimp.utils._axis_str_to_int("Z") == 2
    assert blimp.utils._axis_str_to_int("Y") == 3
    assert blimp.utils._axis_str_to_int("X") == 4
    assert blimp.utils._axis_str_to_int("t") == 0
    assert blimp.utils._axis_str_to_int("c") == 1
    assert blimp.utils._axis_str_to_int("z") == 2
    assert blimp.utils._axis_str_to_int("y") == 3
    assert blimp.utils._axis_str_to_int("x") == 4
    with pytest.raises(ValueError, match=r"Unknown axis : invalid"):
        blimp.utils._axis_str_to_int("invalid")
    with pytest.raises(TypeError, match=r"axis must be int or str"):
        blimp.utils._axis_str_to_int(2.2)


def test_axis_int_to_str():
    assert blimp.utils._axis_int_to_str(0) == "T"
    assert blimp.utils._axis_int_to_str(1) == "C"
    assert blimp.utils._axis_int_to_str(2) == "Z"
    assert blimp.utils._axis_int_to_str(3) == "Y"
    assert blimp.utils._axis_int_to_str(4) == "X"
    with pytest.raises(ValueError, match=r"Unknown axis : 5"):
        blimp.utils._axis_int_to_str(5)
    with pytest.raises(TypeError, match=r"axis must be int or str"):
        blimp.utils._axis_int_to_str(2.2)


def test_confirm_array_rank_ndarray():
    # Test with a single numpy array of rank 3
    image = np.random.rand(10, 10, 10)
    blimp.utils.confirm_array_rank(image)
    # Test with a single numpy array of rank 2, using default rank
    image = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(image)
    # Test with a single numpy array of rank 3, passing incorrect rank
    image = np.random.rand(10, 10, 10)
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(image, rank=2)
    # Test with a single numpy array of rank 2, passing correct rank
    image = np.random.rand(10, 10)
    blimp.utils.confirm_array_rank(image, rank=2)


def test_confirm_array_rank_dask_array():
    # Test with a single dask array of rank 3
    image = da.random.random((10, 10, 10), chunks=(5, 5, 5))
    blimp.utils.confirm_array_rank(image)
    # Test with a single dask array of rank 2
    image = da.random.random((10, 10), chunks=(5, 5))
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(image)
    # Test with a single dask array of rank 3, passing incorrect rank
    image = da.random.random((10, 10, 10), chunks=(5, 5, 5))
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(image, rank=2)
    # Test with a single dask array of rank 2, passing correct rank
    image = da.random.random((10, 10), chunks=(5, 5))
    blimp.utils.confirm_array_rank(image, rank=2)


def test_confirm_array_rank_list():
    # Test with a list of numpy arrays of rank 3
    images = [np.random.rand(10, 10, 10) for _ in range(5)]
    blimp.utils.confirm_array_rank(images)
    # Test with a list of numpy arrays of rank 2
    images = [np.random.rand(10, 10) for _ in range(5)]
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(images)
    # Test with a list of numpy arrays of rank 2, passing non-default rank
    images = [np.random.rand(10, 10) for _ in range(5)]
    blimp.utils.confirm_array_rank(images, rank=2)
    # Test with a list of dask arrays of rank 3
    images = [da.random.random((10, 10, 10), chunks=(5, 5, 5)) for _ in range(5)]
    blimp.utils.confirm_array_rank(images)
    # Test with a list of dask arrays of rank 2
    images = [da.random.random((10, 10), chunks=(5, 5)) for _ in range(5)]
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(images)
    # Test with a list of dask arrays of rank 2, passing non-default rank
    images = [da.random.random((10, 10), chunks=(5, 5)) for _ in range(5)]
    blimp.utils.confirm_array_rank(images, rank=2)
    # Test with a mixed list of dask and numpy arrays of rank 3
    images = [da.random.random((10, 10, 10), chunks=(5, 5, 5)), np.random.rand(10, 10, 10)]
    blimp.utils.confirm_array_rank(images)
    # Test with a mixed list of dask and numpy arrays of rank 2
    images = [da.random.random((10, 10), chunks=(5, 5)), np.random.rand(10, 10)]
    with pytest.raises(ValueError):
        blimp.utils.confirm_array_rank(images)


def test_confirm_array_rank_invalid_input():
    # Test with non-array input
    with pytest.raises(TypeError):
        blimp.utils.confirm_array_rank("string")
    with pytest.raises(TypeError):
        blimp.utils.confirm_array_rank([1, 2, 3])
    # Test with list containing non-array input
    with pytest.raises(TypeError):
        blimp.utils.confirm_array_rank(["string", np.random.rand(10, 10, 10)])


def test_check_correct_dimension_order():
    # define input data
    img1 = AICSImage(np.random.rand(100, 100))
    # Test input of single AICSImage
    assert blimp.utils.check_correct_dimension_order(img1) is True
    # Test input of list of AICSImages
    assert blimp.utils.check_correct_dimension_order([img1, img1]) is True
    # Test input of non-AICSImage or list of non-AICSImages
    with pytest.raises(TypeError):
        blimp.utils.check_correct_dimension_order("not an AICSImage")
    with pytest.raises(TypeError):
        blimp.utils.check_correct_dimension_order([img1, "not an AICSImage"])


def test_check_uniform_dimension_sizes():
    # Test case where all AICSImages in list have matching dimension sizes
    images = [AICSImage(np.random.rand(2, 1, 10, 10, 10)) for _ in range(5)]
    assert blimp.utils.check_uniform_dimension_sizes(images) is True

    # Test case where one AICSImage in list has different dimension sizes
    images = [AICSImage(np.random.rand(2, 2, 9, 10, 10)), AICSImage(np.random.rand(2, 2, 10, 10, 10))]
    assert blimp.utils.check_uniform_dimension_sizes(images) is False

    # Test case where one AICSImage in list has different dimension sizes
    # and this axis is omitted
    assert blimp.utils.check_uniform_dimension_sizes(images, omit="Z") is True

    # Test case where one AICSImage in list has different dimension sizes
    # and a different axis is omitted
    assert blimp.utils.check_uniform_dimension_sizes(images, omit="T") is False

    # Test case where input is a single AICSImage
    image = AICSImage(np.random.rand(10, 10, 10))
    assert blimp.utils.check_uniform_dimension_sizes(image) is True

    # Test case where input is not an AICSImage or list of AICSImages
    with pytest.raises(TypeError):
        blimp.utils.check_uniform_dimension_sizes("not an AICSImage or list of AICSImages")
    with pytest.raises(TypeError):
        blimp.utils.check_uniform_dimension_sizes(
            [AICSImage(np.random.rand(10, 10, 10)), "not an AICSImage or list of AICSImages"]
        )

    # Test case where input AICSImages have different dtypes
    with pytest.raises(TypeError):
        blimp.utils.check_uniform_dimension_sizes(
            [AICSImage(np.random.rand(10, 10, 10)), AICSImage(np.random.rand(10.0, 10.0, 10.0))]
        )


@pytest.fixture
def image_list():
    return [
        AICSImage(
            (65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G", "B"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
        AICSImage(
            (65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G", "B"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
        AICSImage(
            (65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G", "B"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
    ]


def test_average_images_valid_input(image_list):
    result = blimp.utils.average_images(image_list)
    assert isinstance(result, AICSImage)
    assert result.dims.C == 3
    assert result.channel_names == ["R", "G", "B"]
    assert result.physical_pixel_sizes == PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6)


def test_average_images_invalid_input_type():
    with pytest.raises(TypeError):
        blimp.utils.average_images([1, 2, 3])


@pytest.fixture
def image_list_invalid_channel_size():
    return [
        AICSImage(
            (65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G", "B"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
        AICSImage(
            (65536 * np.random.rand(2, 2, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
        AICSImage(
            (65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16),
            channel_names=["R", "G", "B"],
            physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
        ),
    ]


def test_average_images_invalid_input_channel_size(image_list_invalid_channel_size):
    with pytest.raises(ValueError):
        blimp.utils.average_images(image_list_invalid_channel_size)


def test_average_images_dtypes(image_list):

    # Test that dtypes do not change
    result = blimp.utils.average_images(image_list)
    assert result.dtype == image_list[0].dtype

    result = blimp.utils.average_images(
        [
            AICSImage((65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.float64)),
            AICSImage((65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.float64)),
        ]
    )
    assert result.dtype == np.float64

    # Test that non-matching dtypes cannot be averaged
    with pytest.raises(TypeError):
        blimp.utils.average_images(
            [
                AICSImage((65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.uint16)),
                AICSImage((65536 * np.random.rand(2, 3, 4, 10, 10)).astype(np.float64)),
            ]
        )


def test_average_images_accuracy():
    zeros = AICSImage(
        np.zeros(shape=(1, 3, 1, 10, 10), dtype=np.uint16),
        channel_names=["R", "G", "B"],
        physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
    )
    tens = AICSImage(
        np.full(shape=(1, 3, 1, 10, 10), fill_value=10, dtype=np.uint16),
        channel_names=["R", "G", "B"],
        physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
    )
    true = AICSImage(
        np.full(shape=(1, 3, 1, 10, 10), fill_value=5, dtype=np.uint16),
        channel_names=["R", "G", "B"],
        physical_pixel_sizes=PhysicalPixelSizes(Z=1e-6, Y=0.65e-6, X=0.65e-6),
    )
    result = blimp.utils.average_images([zeros, tens])
    np.testing.assert_almost_equal(result.get_image_data("TCZYX"), true.get_image_data("TCZYX"))
    assert result.get_image_data("TCZYX").dtype == true.get_image_data("TCZYX").dtype
