"""Tests for harmony_to_ome_ngff.py -- stitching synthetic (see
tests/helpers.py) and real Opera Phenix / Operetta exports into a
whole-plate OME-NGFF store."""
from pathlib import Path
import shutil

from ngio import open_ome_zarr_plate
import numpy as np
import pandas as pd
import pytest
import tifffile

from blimp.preprocessing.harmony_to_ome_ngff import (
    check_harmony_ngff,
    harmony_to_ome_ngff,
    generate_pbs_script_ngff,
    directions_from_orientation_matrix,
    get_field_layout_from_harmony_metadata,
)
from blimp.preprocessing.opera_parse_metadata import (
    get_image_metadata as get_opera_image_metadata,
)

from .helpers import _make_opera_dataset, _make_operetta_dataset

OPERA_ORIENTATION = "[[0.993265,0,0,-98.8],[0,-0.993265,0,9.3],[0,0,1.00,-2.151]]"
OPERETTA_ORIENTATION = "[[0.999908,0.014866,0,-87.6],[0.014967,-0.999607,0,-85.6],[0,0,1.00,-0.034]]"


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (OPERA_ORIENTATION, ("up", "right")),
        (OPERETTA_ORIENTATION, ("up", "right")),
        ("[[-1,0,0,0],[0,1,0,0],[0,0,1,0]]", ("down", "left")),
    ],
)
def test_directions_from_orientation_matrix(matrix, expected):
    assert directions_from_orientation_matrix(matrix) == expected


def _make_dataset(instrument, root, **kwargs):
    """Synthetic export for ``instrument``, returning (images dir, pixels,
    first timepoint id)."""
    if instrument == "opera":
        pixels = _make_opera_dataset(root, **kwargs)
        return root / "images", pixels, 1
    pixels = _make_operetta_dataset(root, **kwargs)
    return root / "Images", pixels, 0


@pytest.fixture
def opera_well_metadata(tmp_path):
    _make_opera_dataset(tmp_path / "measurement")
    image_metadata = get_opera_image_metadata(tmp_path / "measurement" / "images" / "index.xml")
    return image_metadata.query("Row==3 & Col==5")


def test_get_field_layout_from_harmony_metadata(opera_well_metadata):
    layout = get_field_layout_from_harmony_metadata(opera_well_metadata)
    assert (layout.row, layout.column) == ("C", 5)
    # field 1 lies to the right of field 2, with increasing stage-x to the right
    assert layout.field_ids == [1, 2]
    assert layout.offsets == [(0, 16), (0, 0)]
    assert layout.tile_shape == (1, 2, 3, 16, 16)
    assert layout.canvas_shape == (1, 2, 3, 16, 32)
    assert layout.pixel_size_x == pytest.approx(0.1)
    assert layout.pixel_size_z == pytest.approx(0.5)
    assert layout.channel_names == ["DAPI", "GFP"]
    # emission at 550 and 650 nm
    assert layout.channel_colors == ["00FF00", "FF00FF"]
    assert layout.position_names == ["C05_0001", "C05_0002"]


def test_get_field_layout_from_harmony_metadata_direction_override(opera_well_metadata):
    layout = get_field_layout_from_harmony_metadata(opera_well_metadata, x_direction="left")
    assert layout.offsets == [(0, 0), (0, 16)]


def test_get_field_layout_from_harmony_metadata_places_rows_by_stage_y(tmp_path):
    # field 2 is above field 1 (higher stage-y), so is placed first
    _make_opera_dataset(tmp_path / "m", wells=((3, 5),), field_positions=((0.0, 0.0), (0.0, 1e-4)))
    well_metadata = get_opera_image_metadata(tmp_path / "m" / "images" / "index.xml")
    layout = get_field_layout_from_harmony_metadata(well_metadata)
    assert layout.offsets == [(16, 0), (0, 0)]
    assert layout.canvas_shape[3:] == (32, 16)


def test_get_field_layout_from_harmony_metadata_requires_one_well(tmp_path):
    _make_opera_dataset(tmp_path / "m")
    image_metadata = get_opera_image_metadata(tmp_path / "m" / "images" / "index.xml")
    with pytest.raises(ValueError, match="exactly one well"):
        get_field_layout_from_harmony_metadata(image_metadata)


@pytest.mark.parametrize("instrument", ["opera", "operetta"])
def test_harmony_to_ome_ngff_stitches_wells(instrument, tmp_path):
    images_dir, pixels, t0 = _make_dataset(instrument, tmp_path / "measurement")
    plate_path = tmp_path / "out" / "plate.zarr"
    harmony_to_ome_ngff(images_dir, plate_path, instrument, keep_stacks=True, mip=True)

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    assert plate.rows == list("ABCDEFGH")
    assert sorted(plate.wells_paths()) == ["C/05", "C/06"]

    stack = plate.get_image("C", "06", "stack").get_image().get_as_numpy()
    assert stack.shape == (1, 2, 3, 16, 32)
    for c in (1, 2):
        expected = np.stack([pixels[(3, 6, 1, p, c, t0)] for p in (1, 2, 3)])
        np.testing.assert_array_equal(stack[0, c - 1, :, :, 16:], expected)
        expected = np.stack([pixels[(3, 6, 2, p, c, t0)] for p in (1, 2, 3)])
        np.testing.assert_array_equal(stack[0, c - 1, :, :, :16], expected)

    mip = plate.get_image("C", "06", "mip").get_image().get_as_numpy()
    np.testing.assert_array_equal(mip[:, :, 0], stack.max(axis=2))

    assert (tmp_path / "out" / "image_metadata.csv").exists()
    assert (tmp_path / "out" / "plate_metadata.csv").exists()


@pytest.mark.parametrize("instrument", ["opera", "operetta"])
def test_harmony_to_ome_ngff_combines_timepoints(instrument, tmp_path):
    images_dir, pixels, t0 = _make_dataset(
        instrument, tmp_path / "measurement", wells=((2, 2),), field_positions=((0.0, 0.0),), n_timepoints=2
    )
    plate_path = tmp_path / "plate.zarr"
    harmony_to_ome_ngff(images_dir, plate_path, instrument, keep_stacks=True)

    stack = open_ome_zarr_plate(store=str(plate_path)).get_image("B", "02", "stack").get_image().get_as_numpy()
    assert stack.shape == (2, 2, 3, 16, 16)
    np.testing.assert_array_equal(stack[1, 0, 2], pixels[(2, 2, 1, 3, 1, t0 + 1)])


def test_harmony_to_ome_ngff_batches_by_well_and_check_reports_missing(tmp_path):
    images_dir, _, _ = _make_dataset("opera", tmp_path / "measurement")
    plate_path = tmp_path / "plate.zarr"

    harmony_to_ome_ngff(images_dir, plate_path, "opera", n_batches=2, batch_id=0, mip=True)
    assert open_ome_zarr_plate(store=str(plate_path)).wells_paths() == ["C/05"]
    assert check_harmony_ngff(images_dir, plate_path, "opera", mip=True, keep_stacks=False) == {("C06", "mip")}
    assert check_harmony_ngff(images_dir, plate_path, "opera", mip=True, keep_stacks=True) == {
        ("C05", "stack"),
        ("C06", "stack"),
        ("C06", "mip"),
    }

    harmony_to_ome_ngff(images_dir, plate_path, "opera", n_batches=2, batch_id=1, mip=True)
    assert check_harmony_ngff(images_dir, plate_path, "opera", mip=True, keep_stacks=False) == set()


def test_harmony_to_ome_ngff_rejects_unknown_instrument(tmp_path):
    with pytest.raises(ValueError, match="instrument"):
        harmony_to_ome_ngff(tmp_path, tmp_path / "plate.zarr", "operetta-cls")


def test_generate_pbs_script_ngff_formats_template():
    template = (
        "{INPUT_DIR}|{PLATE_PATH}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}|"
        "{MIP}|{KEEP_STACKS}|{SAVE_METADATA_FILES}|{DIRECTIONS}|{PLACEMENT}|{CHANNEL_NAMES}|{CONDA_ENV}"
    )
    kwargs = dict(
        template=template,
        input_dir="/in",
        plate_path="/out/plate.zarr",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=4,
        mip=True,
        keep_stacks=False,
        save_metadata_files=True,
        conda_env="env",
    )
    assert generate_pbs_script_ngff(**kwargs) == (
        "/in|/out/plate.zarr|/log|z1234567|a@b.com|4|#PBS -J 0-3|${PBS_ARRAY_INDEX}|"
        "--mip||--save_metadata_files||grid||env"
    )
    assert generate_pbs_script_ngff(
        **kwargs, y_direction="down", x_direction="left", placement="exact", channel_names=["A", "B"]
    ).endswith("|-y down -x left|exact|--channel_names A B|env")


# --- reference data ---


@pytest.fixture
def opera_reference_images_dir(test_data, tmp_path):
    measurement = "7ff28d40-ab05-4813-b326-50cfde35c09d"
    src = Path(test_data) / "datasets" / "opera_phenix" / measurement
    return Path(shutil.copytree(src, tmp_path / measurement)) / "images"


@pytest.fixture
def operetta_reference_images_dir(test_data, tmp_path):
    measurement = "20240905_Cycle1_4i_experiment__2024-09-05T20_03_34-Measurement_1"
    src = Path(test_data) / "datasets" / "operetta" / measurement
    return Path(shutil.copytree(src, tmp_path / measurement)) / "Images"


def test_harmony_to_ome_ngff_opera_reference_data(opera_reference_images_dir, tmp_path):
    images_dir = opera_reference_images_dir
    plate_path = tmp_path / "plate.zarr"
    harmony_to_ome_ngff(images_dir, plate_path, "opera", keep_stacks=True, mip=True)

    plate = open_ome_zarr_plate(store=str(plate_path))
    assert plate.meta.plate.name == "hipsctest"
    image = plate.get_image("C", "05", "stack").get_image()
    assert image.pixel_size.x == pytest.approx(0.0941867, rel=1e-5)
    assert image.pixel_size.z == pytest.approx(3.5, rel=1e-3)
    stack = image.get_as_numpy()
    assert stack.shape == (1, 1, 3, 128, 256)

    # field 2 lies to the left of field 1
    for field_id, x0 in ((2, 0), (1, 128)):
        raw = np.stack(
            [tifffile.imread(images_dir / "r03c05" / f"r03c05f{field_id:02d}p{p:02d}-ch01t01.tiff") for p in (1, 8, 15)]
        )
        np.testing.assert_array_equal(stack[0, 0, :, :, x0 : x0 + 128], raw)

    roi_table = plate.get_image("C", "05", "mip").get_table("FOV_ROI_table")
    assert sorted(roi.name for roi in roi_table.rois()) == ["C05_0001", "C05_0002"]


def test_harmony_to_ome_ngff_operetta_reference_data(operetta_reference_images_dir, tmp_path):
    images_dir = operetta_reference_images_dir
    plate_path = tmp_path / "plate.zarr"
    harmony_to_ome_ngff(images_dir, plate_path, "operetta", mip=True)

    container = open_ome_zarr_plate(store=str(plate_path)).get_image("B", "02", "mip")
    assert container.channel_labels == ["Alexa 568", "Alexa 647", "DAPI"]
    mip = container.get_image().get_as_numpy()
    assert mip.shape == (1, 3, 1, 256, 256)

    # field 4 lies above and to the right of field 1
    for field_id, (y0, x0) in ((4, (0, 128)), (1, (128, 0))):
        for c in (1, 2, 3):
            raw = np.stack(
                [tifffile.imread(images_dir / f"r02c02f{field_id:02d}p{p:02d}-ch{c}sk1fk1fl1.tiff") for p in (1, 8, 14)]
            )
            np.testing.assert_array_equal(mip[0, c - 1, 0, y0 : y0 + 128, x0 : x0 + 128], raw.max(axis=0))
    np.testing.assert_array_equal(mip[..., :128, :128], 0)

    assert pd.read_csv(tmp_path / "image_metadata.csv").shape[0] == 18
