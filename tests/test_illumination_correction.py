from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.utils import equal_dims
from blimp.constants import blimp_config
import blimp.preprocessing.illumination_correction

from .helpers import _ensure_test_data  # noqa: F401, I252
from .helpers import _load_test_data

logger = logging.getLogger(__name__)


def test_IlluminationCorrection_init_empty():
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.timelapse is None
    assert illumination_correction.file_path is None
    assert illumination_correction.file_name is None
    assert illumination_correction.method == "pixel_z_score"
    assert illumination_correction.mean_image is None
    assert illumination_correction.std_image is None
    assert illumination_correction.mean_mean_image is None
    assert illumination_correction.mean_std_image is None
    assert illumination_correction.correctors is None
    assert illumination_correction.is_smoothed is False


def test_IlluminationCorrection_invalid_method():
    with pytest.raises(ValueError):
        blimp.preprocessing.illumination_correction.IlluminationCorrection(method="invalid_method")


def test_IlluminationCorrection_invalid_timelapse():
    images = _load_test_data("illumination_correction")
    with pytest.raises(ValueError):
        blimp.preprocessing.illumination_correction.IlluminationCorrection(reference_images=images, timelapse=None)


def test_IlluminationCorrection_fit_invalid_images():
    images = _load_test_data("illumination_correction")
    images[0] = np.random.rand(10, 10)  # Invalid image type
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        method="pixel_z_score", timelapse=False
    )
    with pytest.raises(TypeError):
        illumination_correction.fit(images)


def test_IlluminationCorrection_fit_non_uniform_images():
    images = _load_test_data("illumination_correction")
    images[0] = AICSImage(images[0].data[:, :, :, :100, :100])  # Non-uniform dimensions
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        method="pixel_z_score", timelapse=False
    )
    with pytest.raises(ValueError):
        illumination_correction.fit(images)


def test_IlluminationCorrection_smooth_invalid_method():
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False, method="basic"
    )
    with pytest.raises(NotImplementedError):
        illumination_correction.smooth(sigma=3)


def test_IlluminationCorrection_save_invalid():
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    with pytest.raises(RuntimeError):
        illumination_correction.save("/invalid/path")


def test_IlluminationCorrection_load_invalid():
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    with pytest.raises(FileNotFoundError):
        illumination_correction.load("/invalid/path")


def test_IlluminationCorrection_load_invalid_type():
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    with pytest.raises(TypeError):
        illumination_correction.load(123)


def test_correct_illumination_invalid_input():
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    with pytest.raises(TypeError):
        blimp.preprocessing.illumination_correction.correct_illumination(
            images="invalid_input", illumination_correction=illumination_correction
        )


def test_correct_illumination_invalid_dimension_order():
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    numpy_input = images[0].get_image_data("CZYX")
    with pytest.raises(ValueError):
        blimp.preprocessing.illumination_correction._correct_illumination(
            image=numpy_input, illumination_correction=illumination_correction, dimension_order_in=None
        )


def test_pixel_z_score():
    images = [np.random.rand(10, 10) for _ in range(5)]
    original = images[0]
    mean_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)
    mean_mean_image = np.mean(mean_image)
    mean_std_image = np.mean(std_image)

    corrected = blimp.preprocessing.illumination_correction.pixel_z_score(
        original, mean_image, std_image, mean_mean_image, mean_std_image
    )
    assert corrected.shape == original.shape
    assert corrected.dtype == original.dtype


def test_correct_illumination_list_input(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    numpy_inputs = [image.get_image_data("CZYX") for image in images]
    numpy_outputs = blimp.preprocessing.illumination_correction.correct_illumination(
        images=numpy_inputs, illumination_correction=illumination_correction, dimension_order_in="CZYX"
    )
    assert isinstance(numpy_outputs, list)
    assert all(isinstance(output, np.ndarray) for output in numpy_outputs)
    assert all(input.shape == output.shape for input, output in zip(numpy_inputs, numpy_outputs))
    assert all(input.dtype == output.dtype for input, output in zip(numpy_inputs, numpy_outputs))


def test_IlluminationCorrection_fit(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        method="pixel_z_score", timelapse=False
    )
    illumination_correction.fit(images)
    assert illumination_correction.mean_image is not None
    assert illumination_correction.std_image is not None
    assert illumination_correction.mean_mean_image is not None
    assert illumination_correction.mean_std_image is not None

    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        method="basic", timelapse=False
    )
    illumination_correction.fit(images)
    assert illumination_correction.correctors is not None


def test_IlluminationCorrection_smooth(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    illumination_correction.smooth(sigma=3)
    assert illumination_correction.is_smoothed is True


def test_IlluminationCorrection_save_load(_ensure_test_data, tmp_path):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    save_path = tmp_path / "illumination_correction.pkl"
    illumination_correction.save(save_path)
    assert save_path.is_file()

    loaded_illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        from_file=save_path
    )
    assert loaded_illumination_correction.timelapse == illumination_correction.timelapse
    assert loaded_illumination_correction.method == illumination_correction.method
    assert equal_dims(loaded_illumination_correction, illumination_correction)


def test_IlluminationCorrection_init_from_reference_images(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    # invalid: string passed
    with pytest.raises(TypeError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=[images[0], "invalid"]
        )
    # invalid: single image passed
    with pytest.raises(TypeError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=images[0]
        )
    # invalid: timelapse not specified
    with pytest.raises(ValueError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=images
        )
    # valid:
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    assert illumination_correction.timelapse is False
    assert illumination_correction.file_path is None
    assert illumination_correction.file_name is None
    assert equal_dims(illumination_correction, images[0])
    assert illumination_correction.method == "pixel_z_score"
    # invalid: one of the AICSImages has a different size
    images[0] = AICSImage(images[0].data[:, :, :, :100, :100])
    with pytest.raises(ValueError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=images
        )


def test_IlluminationCorrection_init_from_reference_image_files(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    image_paths = [
        Path(testdata_config.DATASET_DIR) / "illumination_correction" / "221103_brightfield_488_568_647_1.nd2",
        Path(testdata_config.DATASET_DIR) / "illumination_correction" / "221103_brightfield_488_568_647_2.nd2",
    ]
    # invalid: non-existing file passed
    with pytest.raises(FileNotFoundError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=[image_paths[0], "invalid"]
        )
    # invalid: single image path passed
    with pytest.raises(TypeError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=image_paths[0]
        )
    # invalid: timelapse not specified
    with pytest.raises(ValueError):
        illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
            reference_images=image_paths
        )
    # valid: Path
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=image_paths, timelapse=False
    )
    assert illumination_correction.timelapse is False
    assert illumination_correction.file_path is None
    assert illumination_correction.file_name is None
    assert equal_dims(illumination_correction, AICSImage(image_paths[0]))
    assert illumination_correction.method == "pixel_z_score"
    # valid: str
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=[str(p) for p in image_paths], timelapse=False
    )
    assert illumination_correction.timelapse is False
    assert illumination_correction.file_path is None
    assert illumination_correction.file_name is None
    assert equal_dims(illumination_correction, AICSImage(image_paths[0]))
    assert illumination_correction.method == "pixel_z_score"


def test_IlluminationCorrection_load(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    Path(testdata_config.RESOURCES_DIR) / "illumination_correction.pkl"
    invalid_file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.txt"
    # invalid
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    with pytest.raises(FileNotFoundError):
        illumination_correction.load(invalid_file_path)
    # valid (path input)
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.correctors is None
    # TODO: Fix this test by adding a valid file to the testdata
    # illumination_correction.load(file_path)
    # assert illumination_correction.correctors is not None
    # valid (str input)
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.correctors is None
    # TODO: Fix this test by adding a valid file to the testdata
    # illumination_correction.load(str(file_path))
    # assert illumination_correction.correctors is not None


def test_IlluminationCorrection_file_path_setter(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.pkl"
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.file_name is None
    assert illumination_correction.file_name is None
    illumination_correction.file_path = file_path
    assert illumination_correction.file_path == file_path
    assert illumination_correction.file_name == file_path.name
    illumination_correction.file_path = str(file_path)
    assert illumination_correction.file_path == file_path
    assert illumination_correction.file_name == file_path.name
    # test loading from file after defining using the setter
    assert illumination_correction.correctors is None
    # TODO: Fix this test by adding a valid file to the testdata
    # illumination_correction.load()
    # assert illumination_correction.correctors is not None


def test_correct_illumination(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    # valid numpy input
    numpy_input = images[0].get_image_data("CZYX")
    numpy_output = blimp.preprocessing.illumination_correction._correct_illumination(
        image=numpy_input, illumination_correction=illumination_correction, dimension_order_in="CZYX"
    )
    assert isinstance(numpy_output, np.ndarray)
    assert numpy_input.shape == numpy_output.shape
    assert numpy_input.dtype == numpy_output.dtype

    # valid dask input
    dask_input = images[0].get_image_dask_data("CXY")
    dask_output = blimp.preprocessing.illumination_correction._correct_illumination(
        image=dask_input, illumination_correction=illumination_correction, dimension_order_in="CXY"
    )
    assert isinstance(dask_output, np.ndarray)
    assert dask_input.shape == dask_output.shape
    assert dask_input.dtype == dask_output.dtype

    # valid AICSImage input
    AICSImage_input = images[0]
    AICSImage_output = blimp.preprocessing.illumination_correction._correct_illumination(
        image=AICSImage_input, illumination_correction=illumination_correction
    )
    assert isinstance(AICSImage_output, AICSImage)
    assert AICSImage_input.shape == AICSImage_output.shape
    assert AICSImage_input.dtype == AICSImage_output.dtype
    assert AICSImage_input.physical_pixel_sizes == AICSImage_output.physical_pixel_sizes
    assert AICSImage_input.channel_names == AICSImage_output.channel_names


def test_IlluminationCorrection_copy(_ensure_test_data):
    images = _load_test_data("illumination_correction")
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    illumination_correction_copy = illumination_correction.copy()
    assert illumination_correction_copy.timelapse == illumination_correction.timelapse
    assert illumination_correction_copy.method == illumination_correction.method
    assert illumination_correction_copy.dims == illumination_correction.dims
    assert illumination_correction_copy.correctors == illumination_correction.correctors
    assert illumination_correction_copy.mean_image == illumination_correction.mean_image
    assert illumination_correction_copy.std_image == illumination_correction.std_image
    assert illumination_correction_copy.mean_mean_image == illumination_correction.mean_mean_image
    assert illumination_correction_copy.mean_std_image == illumination_correction.mean_std_image
    assert illumination_correction_copy.is_smoothed == illumination_correction.is_smoothed
    assert illumination_correction_copy.file_path == None
    assert illumination_correction_copy.file_name == None
