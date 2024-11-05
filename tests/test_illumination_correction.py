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
    # invalid: one of the AICSImages has a different size
    # TODO
    # valid:
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(
        reference_images=images, timelapse=False
    )
    assert illumination_correction.timelapse is False
    assert illumination_correction.file_path is None
    assert illumination_correction.file_name is None
    assert equal_dims(illumination_correction, images[0])
    assert illumination_correction.method == "pixel_z_score"


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


def test_IlluminationCorrection_init_from_file(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.pkl"
    invalid_file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.txt"
    with pytest.raises(FileNotFoundError):
        blimp.preprocessing.illumination_correction.IlluminationCorrection(from_file=invalid_file_path)
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection(from_file=file_path)
    assert isinstance(illumination_correction.file_path, Path)
    assert illumination_correction.file_path == file_path


def test_IlluminationCorrection_load(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.pkl"
    invalid_file_path = Path(testdata_config.RESOURCES_DIR) / "illumination_correction.txt"
    # valid (path input)
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.correctors is None
    illumination_correction.load(file_path)
    assert illumination_correction.correctors is not None
    # invalid
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    with pytest.raises(FileNotFoundError):
        illumination_correction.load(invalid_file_path)
    # valid (str input)
    illumination_correction = blimp.preprocessing.illumination_correction.IlluminationCorrection()
    assert illumination_correction.correctors is None
    illumination_correction.load(str(file_path))
    assert illumination_correction.correctors is not None


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
    illumination_correction.load()
    assert illumination_correction.correctors is not None


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
