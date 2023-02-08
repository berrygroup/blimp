from typing import Tuple
from pathlib import Path
import logging

from itk import ParameterObject
from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.constants import blimp_config
import blimp.preprocessing.registration

from .helpers import _ensure_test_data  # noqa: F401, I252
from .helpers import _load_test_data

logger = logging.getLogger(__name__)


def test_TransformationParameters_init_from_object():
    parameter_object = ParameterObject.New()
    params = blimp.preprocessing.registration.TransformationParameters(from_object=parameter_object)
    assert params.from_object is True
    assert params.parameter_object == parameter_object


def test_TransformationParameters_init_from_file(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "affine.txt"
    # test Path input
    params = blimp.preprocessing.registration.TransformationParameters(from_file=file_path)
    assert params.from_file == str(file_path)
    assert params.parameter_object.GetNumberOfParameterMaps() == 1
    # test str input
    params = blimp.preprocessing.registration.TransformationParameters(from_file=str(file_path))
    assert params.from_file == str(file_path)
    assert params.parameter_object.GetNumberOfParameterMaps() == 1


def test_TransformationParameters_init_transformation_mode():
    transformation_mode = "translation"
    resolutions = 1
    params = blimp.preprocessing.registration.TransformationParameters(
        transformation_mode=transformation_mode, resolutions=resolutions
    )
    assert params.transformation_mode == transformation_mode
    assert params.parameter_object.GetNumberOfParameterMaps() == 1

    transformation_mode = "rigid"
    resolutions = 2
    params = blimp.preprocessing.registration.TransformationParameters(
        transformation_mode=transformation_mode, resolutions=resolutions
    )
    assert params.transformation_mode == transformation_mode
    assert params.parameter_object.GetNumberOfParameterMaps() == 1

    transformation_mode = "affine"
    resolutions = 3
    params = blimp.preprocessing.registration.TransformationParameters(
        transformation_mode=transformation_mode, resolutions=resolutions
    )
    assert params.transformation_mode == transformation_mode
    assert params.parameter_object.GetNumberOfParameterMaps() == 1


def test_TransformationParameters_init_empty():
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.TransformationParameters()


def test_TransformationParameters_init_resources():
    with pytest.raises(NotImplementedError):
        params = blimp.preprocessing.registration.TransformationParameters(from_resources="rigid_20x_operetta.txt")


def test_TransformationParameters_save(_ensure_test_data):
    # Test saving
    testdata_config = blimp_config.get_data_config("testdata")
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    file_name = Path(testdata_config.RESULTS_DIR) / "rigid.txt"
    params = blimp.preprocessing.registration.TransformationParameters(transformation_mode="rigid")
    params.save(file_name)
    assert Path(file_name).exists()
    # Test reloading
    assert isinstance(
        blimp.preprocessing.registration.TransformationParameters(from_file=file_name),
        blimp.preprocessing.registration.TransformationParameters,
    )


def test_TransformationParameters_init_unknown_transformation_mode():
    transformation_mode = "unknown"
    with pytest.raises(ValueError):
        params = blimp.preprocessing.registration.TransformationParameters(transformation_mode=transformation_mode)


def test_load_operetta_cls_multiplex_images(_ensure_test_data):
    images = _load_test_data("operetta_cls_multiplex")
    for image in images:
        assert isinstance(image, AICSImage)
        assert image.dims.order == "TCZYX"
    assert images[0].shape == (1, 2, 1, 2160, 2160)
    assert images[1].shape == (1, 3, 1, 2160, 2160)


def test_register_2D_valid_input(_ensure_test_data):
    images = _load_test_data("operetta_cls_multiplex")
    fixed_np = images[0].get_image_data("YX", T=0, C=0, Z=0)
    moving_np = images[1].get_image_data("YX", T=0, C=0, Z=0)
    fixed_da = images[0].get_image_dask_data("YX", T=0, C=0, Z=0)
    moving_da = images[1].get_image_dask_data("YX", T=0, C=0, Z=0)
    settings = blimp.preprocessing.registration.TransformationParameters("translation")
    # Test numpy input
    registered_np, parameters_np = blimp.preprocessing.registration.register_2D(
        fixed=fixed_np, moving=moving_np, settings=settings
    )
    assert isinstance(registered_np, np.ndarray)
    assert registered_np.shape == fixed_np.shape
    assert isinstance(parameters_np, blimp.preprocessing.registration.TransformationParameters)
    # Test dask input
    registered_da, parameters_da = blimp.preprocessing.registration.register_2D(
        fixed=fixed_da, moving=moving_da, settings=settings
    )
    assert isinstance(registered_da, np.ndarray)
    assert registered_da.shape == fixed_da.shape
    assert isinstance(parameters_da, blimp.preprocessing.registration.TransformationParameters)
    # Test whether dask and numpy inputs are equivalent
    np.testing.assert_almost_equal(registered_da, registered_np)
    # Test whether registration modified the input image
    np.testing.assert_raises(AssertionError, np.testing.assert_almost_equal, registered_np, moving_np)


def test_register_2D_parameters_only():
    fixed = np.random.rand(10, 10)
    moving = np.random.rand(10, 10)
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    parameters = blimp.preprocessing.registration.register_2D(fixed, moving, settings, parameters_only=True)
    assert isinstance(parameters, blimp.preprocessing.registration.TransformationParameters)


def test_register_2D_preserve_dtype():
    fixed_float = np.random.rand(10, 10).astype(np.float32)
    moving_float = np.random.rand(10, 10).astype(np.float32)
    fixed_int8 = (255 * np.random.rand(10, 10) - 128).astype(np.int8)
    moving_int8 = (255 * np.random.rand(10, 10) - 128).astype(np.int8)
    fixed_uint16 = (65535 * np.random.rand(10, 10)).astype(np.uint16)
    moving_uint16 = (65535 * np.random.rand(10, 10)).astype(np.uint16)
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    registered, parameters = blimp.preprocessing.registration.register_2D(fixed_float, moving_float, settings)
    assert registered.dtype == fixed_float.dtype
    registered, parameters = blimp.preprocessing.registration.register_2D(fixed_int8, moving_int8, settings)
    assert registered.dtype == fixed_int8.dtype
    registered, parameters = blimp.preprocessing.registration.register_2D(fixed_uint16, moving_uint16, settings)
    assert registered.dtype == fixed_uint16.dtype
    # Test whether non-matching data types raise the correct error
    with pytest.raises(TypeError):
        registered, parameters = blimp.preprocessing.registration.register_2D(fixed_float, moving_uint16, settings)
    with pytest.raises(TypeError):
        registered, parameters = blimp.preprocessing.registration.register_2D(fixed_int8, moving_uint16, settings)


def test_register_2D_invalid_input():
    fixed = np.random.rand(10, 10)
    moving = np.random.rand(5, 5)
    moving_rank_3 = np.random.rand(1, 10, 10)
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.register_2D(fixed, moving, settings)
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.register_2D(fixed, moving_rank_3, settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.register_2D(fixed, "invalid", settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.register_2D(fixed, fixed, "invalid")


def test_transform_2D_valid_input(_ensure_test_data):
    # Load images and transformation_settings
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "affine.txt"
    images = _load_test_data("operetta_cls_multiplex")
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)
    parameters = blimp.preprocessing.registration.TransformationParameters(from_file=file_path)
    transformed = blimp.preprocessing.registration.transform_2D(moving, parameters)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == moving.shape


def test_transform_2D_non_matching_sizes(_ensure_test_data):
    # Load images and transformation_settings
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "affine.txt"
    images = _load_test_data("operetta_cls_multiplex")
    # crop input image so that it does not match the
    # size of the loaded transformation settings
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)[:-100, :-100]
    parameters = blimp.preprocessing.registration.TransformationParameters(from_file=file_path)
    with pytest.raises(RuntimeError):
        blimp.preprocessing.registration.transform_2D(moving, parameters)


def test_transform_2D_invalid_input():
    moving = np.random.rand(5, 5)
    moving_rank_3 = np.random.rand(1, 10, 10)
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.transform_2D(moving_rank_3, settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.transform_2D("invalid", settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.transform_2D(moving, "invalid")


def test_compare_register_2D_and_transform_2D(_ensure_test_data):
    images = _load_test_data("operetta_cls_multiplex")
    fixed = images[0].get_image_data("YX", T=0, C=0, Z=0)
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)
    settings = blimp.preprocessing.registration.TransformationParameters("rigid")
    registered_1, parameters = blimp.preprocessing.registration.register_2D(
        fixed=fixed, moving=moving, settings=settings
    )
    registered_2 = blimp.preprocessing.registration.transform_2D(moving=moving, parameters=parameters)
    np.testing.assert_almost_equal(registered_1, registered_2)


def test_register_2D_accuracy(_ensure_test_data):
    images = _load_test_data("registration_tests")
    original = images[0].get_image_data("YX")
    translated = images[1].get_image_data("YX")
    rotated = images[2].get_image_data("YX")
    rotated_registered_validated = images[3].get_image_data("YX")

    # Test if translation of a pure translation can recover the original
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    translated_registered, parameters = blimp.preprocessing.registration.register_2D(original, translated, settings)
    original_difference = original[200:-200, 200:-200].astype(np.float64) - translated[200:-200, 200:-200].astype(
        np.float64
    )
    registered_difference = original[200:-200, 200:-200].astype(np.float64) - translated_registered[
        200:-200, 200:-200
    ].astype(np.float64)
    assert np.mean(abs(original_difference)) > 1
    assert np.mean(abs(registered_difference)) < 1

    # Test if rigid registration of a rotation matches previously validated image
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="rigid")
    rotated_registered, parameters = blimp.preprocessing.registration.register_2D(original, rotated, settings)
    np.testing.assert_almost_equal(rotated_registered, rotated_registered_validated)

    # Check that translation fails to recover the previously validated image
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    rotated_registered, parameters = blimp.preprocessing.registration.register_2D(original, rotated, settings)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_almost_equal, rotated_registered, rotated_registered_validated
    )


def test_register_2D_fast_valid_input(_ensure_test_data):
    images = _load_test_data("operetta_cls_multiplex")
    fixed_np = images[0].get_image_data("YX", T=0, C=0, Z=0)
    moving_np = images[1].get_image_data("YX", T=0, C=0, Z=0)
    fixed_da = images[0].get_image_dask_data("YX", T=0, C=0, Z=0)
    moving_da = images[1].get_image_dask_data("YX", T=0, C=0, Z=0)

    # Test numpy input
    registered_np, parameters_np = blimp.preprocessing.registration.register_2D_fast(fixed=fixed_np, moving=moving_np)
    assert isinstance(registered_np, np.ndarray)
    assert registered_np.shape == fixed_np.shape
    assert isinstance(parameters_np, tuple)
    # Test dask input
    registered_da, parameters_da = blimp.preprocessing.registration.register_2D_fast(fixed=fixed_da, moving=moving_da)
    assert isinstance(registered_da, np.ndarray)
    assert registered_da.shape == fixed_da.shape
    assert isinstance(parameters_da, tuple)
    # Test whether dask and numpy inputs are equivalent
    np.testing.assert_almost_equal(registered_da, registered_np)
    # Test whether registration modified the input image
    np.testing.assert_raises(AssertionError, np.testing.assert_almost_equal, registered_np, moving_np)


def test_register_2D_fast_parameters_only():
    fixed = np.random.rand(10, 10)
    moving = np.random.rand(10, 10)
    parameters = blimp.preprocessing.registration.register_2D_fast(fixed, moving, parameters_only=True)
    assert isinstance(parameters, tuple)


def test_register_2D_fast_preserve_dtype():
    fixed_float = np.random.rand(10, 10).astype(np.float32)
    moving_float = np.random.rand(10, 10).astype(np.float32)
    fixed_int8 = (255 * np.random.rand(10, 10) - 128).astype(np.int8)
    moving_int8 = (255 * np.random.rand(10, 10) - 128).astype(np.int8)
    fixed_uint16 = (65535 * np.random.rand(10, 10)).astype(np.uint16)
    moving_uint16 = (65535 * np.random.rand(10, 10)).astype(np.uint16)
    registered, parameters = blimp.preprocessing.registration.register_2D_fast(fixed_float, moving_float)
    assert registered.dtype == fixed_float.dtype
    registered, parameters = blimp.preprocessing.registration.register_2D_fast(fixed_int8, moving_int8)
    assert registered.dtype == fixed_int8.dtype
    registered, parameters = blimp.preprocessing.registration.register_2D_fast(fixed_uint16, moving_uint16)
    assert registered.dtype == fixed_uint16.dtype
    # Test whether non-matching data types raise the correct error
    with pytest.raises(TypeError):
        registered, parameters = blimp.preprocessing.registration.register_2D_fast(fixed_float, moving_uint16)
    with pytest.raises(TypeError):
        registered, parameters = blimp.preprocessing.registration.register_2D_fast(fixed_int8, moving_uint16)


def test_register_2D_fast_invalid_input():
    fixed = np.random.rand(10, 10)
    moving = np.random.rand(5, 5)
    moving_rank_3 = np.random.rand(1, 10, 10)
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.register_2D_fast(fixed, moving)
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.register_2D_fast(fixed, moving_rank_3)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.register_2D_fast(fixed, "invalid")


def test_transform_2D_fast_valid_input(_ensure_test_data):
    # Load images and transformation_settings
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "affine.txt"
    images = _load_test_data("operetta_cls_multiplex")
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)
    parameters = blimp.preprocessing.registration.TransformationParameters(from_file=file_path)
    transformed = blimp.preprocessing.registration.transform_2D(moving, parameters)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == moving.shape


def test_transform_2D_fast_non_matching_sizes(_ensure_test_data):
    # Load images and transformation_settings
    testdata_config = blimp_config.get_data_config("testdata")
    file_path = Path(testdata_config.RESOURCES_DIR) / "affine.txt"
    images = _load_test_data("operetta_cls_multiplex")
    # crop input image so that it does not match the
    # size of the loaded transformation settings
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)[:-100, :-100]
    parameters = blimp.preprocessing.registration.TransformationParameters(from_file=file_path)
    with pytest.raises(RuntimeError):
        blimp.preprocessing.registration.transform_2D(moving, parameters)


def test_transform_2D_fast_invalid_input():
    moving = np.random.rand(5, 5)
    moving_rank_3 = np.random.rand(1, 10, 10)
    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.transform_2D_fast(moving_rank_3, settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.transform_2D_fast("invalid", settings)
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.transform_2D_fast(moving, "invalid")
    with pytest.raises(TypeError):
        blimp.preprocessing.registration.transform_2D_fast(moving, (1.2, "string"))


def test_compare_register_2D_fast_and_transform_2D_fast(_ensure_test_data):
    images = _load_test_data("operetta_cls_multiplex")
    fixed = images[0].get_image_data("YX", T=0, C=0, Z=0)
    moving = images[1].get_image_data("YX", T=0, C=0, Z=0)
    registered_1, parameters = blimp.preprocessing.registration.register_2D_fast(
        fixed=fixed,
        moving=moving,
    )
    registered_2 = blimp.preprocessing.registration.transform_2D_fast(moving=moving, parameters=parameters)
    np.testing.assert_almost_equal(registered_1, registered_2)


def test_register_2D_fast_accuracy(_ensure_test_data):
    images = _load_test_data("registration_tests")
    original = images[0].get_image_data("YX")
    translated = images[1].get_image_data("YX")

    # Test if translation of a pure translation can recover the original
    translated_registered, parameters = blimp.preprocessing.registration.register_2D_fast(original, translated)
    original_difference = original[200:-200, 200:-200].astype(np.float64) - translated[200:-200, 200:-200].astype(
        np.float64
    )
    registered_difference = original[200:-200, 200:-200].astype(np.float64) - translated_registered[
        200:-200, 200:-200
    ].astype(np.float64)
    assert np.mean(abs(original_difference)) > 1
    assert np.mean(abs(registered_difference)) < 1


def test_calculate_shifts_elastix(_ensure_test_data):
    test_images = _load_test_data("operetta_cls_multiplex")
    test_images = test_images + test_images
    test_settings = blimp.preprocessing.registration.TransformationParameters("affine")

    # Test that the function returns the correct output type
    result = blimp.preprocessing.registration._calculate_shifts_elastix(test_images, 0, 0, test_settings)
    assert all(isinstance(n, blimp.preprocessing.registration.TransformationParameters) for n in result)


def test_calculate_shifts_image_registration(_ensure_test_data):
    test_images = _load_test_data("operetta_cls_multiplex")
    test_images += test_images

    # Test that the function returns the correct output type
    result = blimp.preprocessing.registration._calculate_shifts_image_registration(test_images, 0, 0)
    assert all(isinstance(n, Tuple) for n in result)


def test_calculate_shifts_errors(_ensure_test_data):
    test_images = _load_test_data("operetta_cls_multiplex")
    test_images += test_images

    # Test valid
    result_parameters = blimp.preprocessing.registration.calculate_shifts(
        test_images,
        1,
        1,
    )

    # Test out-of-range for reference channel not found
    with pytest.raises(IndexError):
        result_parameters = blimp.preprocessing.registration.calculate_shifts(
            test_images,
            2,
            0,
        )

    # Test out-of-range for cycle not found
    with pytest.raises(IndexError):
        result_parameters = blimp.preprocessing.registration.calculate_shifts(
            test_images,
            0,
            4,
        )

    # Test negative channel
    with pytest.raises(IndexError):
        result_parameters = blimp.preprocessing.registration.calculate_shifts(
            test_images,
            -1,
            0,
        )

    # Test negative cycle
    with pytest.raises(IndexError):
        result_parameters = blimp.preprocessing.registration.calculate_shifts(
            test_images,
            0,
            -1,
        )

    # Test that the function returns a ValueError for unknown library request
    with pytest.raises(ValueError):
        result_parameters = blimp.preprocessing.registration.calculate_shifts(test_images, 0, 0, lib="elast")


def test_calculate_shifts_output_types(_ensure_test_data):
    test_images = _load_test_data("operetta_cls_multiplex")
    test_images += test_images

    # Test returns the correct output types
    result_parameters = blimp.preprocessing.registration.calculate_shifts(test_images, 0, 0, lib="elastix")
    assert all(isinstance(n, blimp.preprocessing.registration.TransformationParameters) for n in result_parameters)

    result_parameters = blimp.preprocessing.registration.calculate_shifts(test_images, 0, 0, lib="image_registration")
    assert all(isinstance(n, Tuple) for n in result_parameters)

    # check correct output dimensions
    assert len(result_parameters) == len(test_images)


def test_get_crop_mask_from_transformation_parameters(_ensure_test_data):
    images = _load_test_data("registration_tests")
    original = images[0].get_image_data("YX")
    translated = images[1].get_image_data("YX")

    settings = blimp.preprocessing.registration.TransformationParameters(transformation_mode="translation")
    original_registered, null_parameters = blimp.preprocessing.registration.register_2D(original, original, settings)
    translated_registered, parameters = blimp.preprocessing.registration.register_2D(original, translated, settings)

    # Test case 1: input is a list of TransformationParameters
    parameters_list = [null_parameters, parameters]
    expected_output = np.ones(shape=original.shape, dtype=np.bool_)
    expected_output[-50:, :] = False
    expected_output[:, -20:] = False
    output = blimp.preprocessing.registration._get_crop_mask_from_transformation_parameters(parameters=parameters_list)
    assert np.array_equal(output, expected_output)

    # Test case 2: input is a list of (y,x) shifts
    parameters_list = [(0, 0), (10, 10), (0, -5)]
    shape = (20, 20)
    expected_output = np.ones(shape=shape, dtype=np.bool_)
    expected_output[10:, :] = False
    expected_output[:, 10:] = False
    expected_output[:5, :] = False
    with pytest.raises(ValueError):
        blimp.preprocessing.registration._get_crop_mask_from_transformation_parameters(parameters=parameters_list)
    output = blimp.preprocessing.registration._get_crop_mask_from_transformation_parameters(
        parameters=parameters_list, shape=shape
    )
    assert np.array_equal(output, expected_output)

    # Test case 3: input is a mixed list of TransformationParameters and (y,x) shifts
    parameters_list = [blimp.preprocessing.registration.TransformationParameters(transformation_mode="rigid"), (10, 20)]
    with pytest.raises(TypeError):
        blimp.preprocessing.registration._get_crop_mask_from_transformation_parameters(
            parameters=parameters_list, shape=shape
        )

    # Test case 4: input is not a list
    with pytest.raises(TypeError):
        blimp.preprocessing.registration._get_crop_mask_from_transformation_parameters(parameters=settings)


def test_apply_shifts_with_elastix_library(_ensure_test_data):
    test_images = _load_test_data("operetta_cls_multiplex")

    # Test valid
    transformation_parameters = blimp.preprocessing.registration.calculate_shifts(
        test_images,
        0,
        0,
    )
    # crop False
    result = blimp.preprocessing.registration.apply_shifts(
        test_images, transformation_parameters, lib="elastix", crop=False
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AICSImage) for r in result)
    assert result[0].dims.Z == 1
    assert result[0].shape == test_images[0].shape
    # crop True
    result = blimp.preprocessing.registration.apply_shifts(
        test_images, transformation_parameters, lib="elastix", crop=True
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AICSImage) for r in result)
    assert result[0].dims.Z == 1
    assert result[0].shape != test_images[0].shape

    # unrecognised library
    with pytest.raises(NotImplementedError):
        blimp.preprocessing.registration.apply_shifts(
            test_images, transformation_parameters, lib="unrecognized", crop=True
        )

    # wrong transformation parameter type
    with pytest.raises(ValueError):
        blimp.preprocessing.registration.apply_shifts(
            test_images, transformation_parameters=[(0, 0), (2, 2)], lib="elastix", crop=True
        )


def test_apply_shifts_with_image_registration_library():
    test_images = _load_test_data("operetta_cls_multiplex")

    # Test valid
    transformation_parameters = blimp.preprocessing.registration.calculate_shifts(
        test_images, 0, 0, lib="image_registration"
    )
    # crop False
    result = blimp.preprocessing.registration.apply_shifts(
        test_images, transformation_parameters, lib="image_registration", crop=False
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AICSImage) for r in result)
    assert result[0].dims.Z == 1
    assert result[0].shape == test_images[0].shape
    # crop True
    result = blimp.preprocessing.registration.apply_shifts(
        test_images, transformation_parameters, lib="image_registration", crop=True
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AICSImage) for r in result)
    assert result[0].dims.Z == 1
    assert result[0].shape != test_images[0].shape


def test_apply_shifts_with_non_uniform_dimension_sizes():
    images = [AICSImage(np.zeros((1, 1, 1, 2, 3))), AICSImage(np.zeros((1, 1, 1, 2, 4)))]
    transformation_parameters = [(0, 0), (2, 2)]

    with pytest.raises(ValueError):
        blimp.preprocessing.registration.apply_shifts(
            images, transformation_parameters, lib="image_registration", crop=False
        )


def test_apply_shifts_with_3d_images():
    images = [AICSImage(np.zeros((1, 1, 10, 2, 3))), AICSImage(np.zeros((1, 1, 10, 2, 3)))]
    transformation_parameters = [(0, 0), (0, 0)]

    with pytest.raises(NotImplementedError):
        blimp.preprocessing.registration.apply_shifts(
            images, transformation_parameters, lib="image_registration", crop=False
        )


# TODO: implement quantification check after elastix transformation
#    def test_quantification_from_transformed_images(_ensure_test_data):
#        test_images = _load_test_data("operetta_cls_multiplex")
#        nuclei =
