from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.constants import blimp_config
from blimp.processing.segment import segment_nuclei_cellpose

logger = logging.getLogger(__name__)


def test_segment_nuclei_cellpose_basic(_ensure_test_data, cellpose_models):
    """Test basic segmentation with cellpose on operetta multiplex data."""
    testdata_config = blimp_config.get_data_config("testdata")
    dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"

    # Load test image (cycle 01, channel 0 should be nuclei)
    test_image_path = dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff"
    logger.info(f"Loading test image from {test_image_path}")

    test_image = AICSImage(test_image_path)

    # Crop to 500x500 pixels to speed up test
    cropped_data = test_image.data[:, :, :, :500, :500]
    test_image = AICSImage(
        cropped_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    # Verify it's 2D (Z=1)
    assert test_image.dims.Z == 1, "Test image should be 2D"
    assert test_image.dims.order == "TCZYX"

    # Run segmentation on channel 0 (nuclei)
    logger.info("Running cellpose segmentation on channel 0")
    segmentation = segment_nuclei_cellpose(
        intensity_image=test_image,
        nuclei_channel=0,
        diameter=50,  # Approximate nuclear diameter for this dataset
        gpu=False,
    )

    # Verify output format
    assert isinstance(segmentation, AICSImage)
    assert segmentation.dims.order == "TCZYX"
    assert segmentation.dims.T == test_image.dims.T
    assert segmentation.dims.Z == 1
    assert segmentation.dims.Y == test_image.dims.Y
    assert segmentation.dims.X == test_image.dims.X
    assert segmentation.dims.C == 1
    assert segmentation.channel_names == ["Nuclei"]

    # Verify that segmentation produced some objects
    masks = segmentation.get_image_data("YX", T=0, C=0, Z=0)
    num_objects = len(np.unique(masks)) - 1  # Subtract background
    logger.info(f"Segmented {num_objects} nuclei")
    assert num_objects > 0, "Should detect at least one nucleus"

    # Verify masks are integer labels
    assert masks.dtype in [np.int32, np.int64, np.uint16, np.uint32]


def _intensity_image_2D(array):
    """Wrap a single 2D intensity array (YX) into a TCZYX AICSImage."""
    stack = array[np.newaxis, np.newaxis, np.newaxis, :, :]
    return AICSImage(stack, channel_names=["Nuclei"])


def test_segment_nuclei_cellpose_rescale_limits_conflicts_with_normalize_false():
    """rescale_limits combined with normalize=False must raise before any cellpose call."""
    image = _intensity_image_2D(np.zeros((16, 16), dtype=np.uint16))

    with pytest.raises(ValueError, match="disables normalization"):
        segment_nuclei_cellpose(
            intensity_image=image,
            rescale_limits=(0, 1000),
            normalize=False,
        )


def test_segment_nuclei_cellpose_with_rescale_limits(_ensure_test_data, cellpose_models):
    """Explicit rescale_limits should plumb through to cellpose and still segment nuclei."""
    testdata_config = blimp_config.get_data_config("testdata")
    dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"
    test_image_path = dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff"

    test_image = AICSImage(test_image_path)

    # Crop to 500x500 pixels to speed up test
    cropped_data = test_image.data[:, :, :, :500, :500]
    test_image = AICSImage(
        cropped_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    nuclei_pixels = test_image.get_image_data("YX", C=0, T=0, Z=0)
    rescale_limits = tuple(np.percentile(nuclei_pixels, [1.0, 99.0]))

    segmentation = segment_nuclei_cellpose(
        intensity_image=test_image,
        nuclei_channel=0,
        diameter=50,
        rescale_limits=rescale_limits,
        gpu=False,
    )

    assert isinstance(segmentation, AICSImage)
    assert segmentation.channel_names == ["Nuclei"]

    masks = segmentation.get_image_data("YX", T=0, C=0, Z=0)
    num_objects = len(np.unique(masks)) - 1
    logger.info(f"Segmented {num_objects} nuclei with explicit rescale_limits")
    assert num_objects > 0, "Should detect at least one nucleus"
    assert masks.dtype in [np.int32, np.int64, np.uint16, np.uint32]


def test_segment_nuclei_cellpose_3d_raises_error(_ensure_test_data):
    """Test that 3D images raise ValueError."""
    testdata_config = blimp_config.get_data_config("testdata")
    dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"
    test_image_path = dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff"

    test_image = AICSImage(test_image_path)

    # Crop to 500x500 pixels to speed up test
    cropped_data = test_image.data[:, :, :, :500, :500]
    test_image = AICSImage(
        cropped_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    # Create a fake 3D image by duplicating Z planes
    fake_3d_data = np.repeat(test_image.data, 3, axis=2)  # Repeat along Z dimension
    fake_3d_image = AICSImage(
        fake_3d_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    assert fake_3d_image.dims.Z > 1, "Fake 3D image should have Z > 1"

    # Should raise ValueError for 3D images
    with pytest.raises(ValueError, match="only supports 2D images"):
        segment_nuclei_cellpose(
            intensity_image=fake_3d_image,
            nuclei_channel=0,
            gpu=False,
        )


def test_segment_nuclei_cellpose_multiple_timepoints(_ensure_test_data, cellpose_models):
    """Test segmentation with multiple timepoints."""
    testdata_config = blimp_config.get_data_config("testdata")
    dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"
    test_image_path = dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff"

    test_image = AICSImage(test_image_path)

    # Crop to 500x500 pixels to speed up test
    cropped_data = test_image.data[:, :, :, :500, :500]
    test_image = AICSImage(
        cropped_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    # Create fake multi-timepoint image
    fake_multitime_data = np.repeat(test_image.data, 3, axis=0)  # Repeat along T dimension
    fake_multitime_image = AICSImage(
        fake_multitime_data,
        channel_names=test_image.channel_names,
        physical_pixel_sizes=test_image.physical_pixel_sizes,
    )

    assert fake_multitime_image.dims.T == 3, "Should have 3 timepoints"

    # Run segmentation
    segmentation = segment_nuclei_cellpose(
        intensity_image=fake_multitime_image,
        nuclei_channel=0,
        diameter=50,
        gpu=False,
    )

    # Verify all timepoints were segmented
    assert segmentation.dims.T == 3

    for t in range(3):
        masks = segmentation.get_image_data("YX", T=t, C=0, Z=0)
        num_objects = len(np.unique(masks)) - 1
        logger.info(f"Timepoint {t}: segmented {num_objects} nuclei")
        assert num_objects > 0, f"Should detect nuclei at timepoint {t}"
