"""Tests for opera_to_ome_tiff.py -- conversion of a synthetic Opera Phenix
export (see tests/helpers.py) and of a small real export from the reference
dataset to OME-TIFF."""
from pathlib import Path
import shutil

from bioio import BioImage
import numpy as np
import pandas as pd
import pytest
import tifffile

from blimp.preprocessing.opera_to_ome_tiff import (
    opera_to_ome_tiff,
    _get_zyx_resolution,
    _remove_TCZ_filename,
    _aggregate_TCZ_metadata,
    opera_regex_filename_pattern,
)
from blimp.preprocessing.opera_parse_metadata import get_image_metadata

from .helpers import _make_opera_dataset


@pytest.mark.parametrize(
    "filename,mip,expected",
    [
        ("r03c05/r03c05f01p01-ch01t01.tiff", False, "r03c05f01.ome.tiff"),
        ("r03c05f12p15-ch02t03.tiff", True, "r03c05f12-mip.ome.tiff"),
        ("Thumbs.db", False, None),
    ],
)
def test_remove_TCZ_filename(filename, mip, expected):
    assert _remove_TCZ_filename(opera_regex_filename_pattern, filename, mip) == expected


@pytest.fixture
def opera_dataset(tmp_path):
    measurement_dir = tmp_path / "measurement"
    pixels = _make_opera_dataset(measurement_dir)
    return measurement_dir, pixels


def test_aggregate_TCZ_metadata_one_row_per_site(opera_dataset):
    measurement_dir, _ = opera_dataset
    image_metadata = get_image_metadata(measurement_dir / "images" / "index.xml")
    aggregated = _aggregate_TCZ_metadata(image_metadata, mip=True)
    assert len(aggregated) == 4
    assert set(aggregated["URL"]) == {
        "r03c05f01-mip.ome.tiff",
        "r03c05f02-mip.ome.tiff",
        "r03c06f01-mip.ome.tiff",
        "r03c06f02-mip.ome.tiff",
    }
    assert (aggregated["NPlanes"] == 3).all()
    assert (aggregated["NChannels"] == 2).all()
    assert "ChannelName" not in aggregated.columns


def test_get_zyx_resolution_in_micrometres(opera_dataset):
    measurement_dir, _ = opera_dataset
    image_metadata = get_image_metadata(measurement_dir / "images" / "index.xml")
    voxel_dimensions = _get_zyx_resolution(image_metadata, 3, 5, 1, 1, 1)
    assert voxel_dimensions.X == pytest.approx(0.1)
    assert voxel_dimensions.Y == pytest.approx(0.1)
    assert voxel_dimensions.Z == pytest.approx(0.5)


def test_get_zyx_resolution_single_plane_has_no_z(tmp_path):
    _make_opera_dataset(tmp_path / "measurement", n_planes=1)
    image_metadata = get_image_metadata(tmp_path / "measurement" / "images" / "index.xml")
    assert _get_zyx_resolution(image_metadata, 3, 5, 1, 1, 1).Z is None


def _expected_stack(pixels, row, col, field_id, n_planes=3, n_channels=2, n_timepoints=1):
    """Expected TCZYX array for one site."""
    return np.stack(
        [
            np.stack(
                [
                    np.stack([pixels[(row, col, field_id, p, c, t)] for p in range(1, n_planes + 1)])
                    for c in range(1, n_channels + 1)
                ]
            )
            for t in range(1, n_timepoints + 1)
        ]
    )


def test_opera_to_ome_tiff_writes_stacks_and_mips(opera_dataset):
    measurement_dir, pixels = opera_dataset
    out_path = measurement_dir / "OME-TIFF"
    opera_to_ome_tiff(
        in_path=measurement_dir / "images",
        out_path=out_path,
        mip=True,
        keep_stacks=True,
    )
    out_path_mip = measurement_dir / "OME-TIFF-MIP"

    assert sorted(p.name for p in out_path.glob("*.ome.tiff")) == [
        "r03c05f01.ome.tiff",
        "r03c05f02.ome.tiff",
        "r03c06f01.ome.tiff",
        "r03c06f02.ome.tiff",
    ]
    assert len(list(out_path_mip.glob("*-mip.ome.tiff"))) == 4

    expected = _expected_stack(pixels, 3, 6, 2)
    stack = BioImage(out_path / "r03c06f02.ome.tiff")
    assert stack.dims.order == "TCZYX"
    assert stack.shape == (1, 2, 3, 16, 16)
    assert list(stack.channel_names) == ["DAPI", "GFP"]
    assert stack.physical_pixel_sizes.Z == pytest.approx(0.5)
    assert stack.physical_pixel_sizes.X == pytest.approx(0.1)
    np.testing.assert_array_equal(stack.data, expected)

    mip = BioImage(out_path_mip / "r03c06f02-mip.ome.tiff")
    assert mip.shape == (1, 2, 1, 16, 16)
    np.testing.assert_array_equal(mip.data, expected.max(axis=2, keepdims=True))


def test_opera_to_ome_tiff_saves_metadata_files(opera_dataset):
    measurement_dir, _ = opera_dataset
    out_path = measurement_dir / "OME-TIFF"
    opera_to_ome_tiff(in_path=measurement_dir / "images", out_path=out_path, mip=True, keep_stacks=False)

    assert not list(out_path.glob("*.ome.tiff"))
    for path in [out_path, measurement_dir / "OME-TIFF-MIP"]:
        metadata = pd.read_csv(path / "image_metadata.csv")
        assert len(metadata) == 4
        assert pd.read_pickle(path / "image_metadata.pkl").shape == metadata.shape
    assert (measurement_dir / "plate_metadata.csv").exists()
    assert (measurement_dir / "image_metadata.csv").exists()


def test_opera_to_ome_tiff_batches_cover_all_sites(opera_dataset):
    measurement_dir, _ = opera_dataset
    out_path_mip = measurement_dir / "OME-TIFF-MIP"
    n_written = []
    for batch_id in range(3):
        opera_to_ome_tiff(
            in_path=measurement_dir / "images",
            out_path=measurement_dir / "OME-TIFF",
            n_batches=3,
            batch_id=batch_id,
            mip=True,
        )
        n_written.append(len(list(out_path_mip.glob("*-mip.ome.tiff"))))
    assert n_written == [2, 4, 4]


def test_opera_to_ome_tiff_combines_timepoints(tmp_path):
    measurement_dir = tmp_path / "measurement"
    pixels = _make_opera_dataset(measurement_dir, wells=((2, 2),), field_positions=((0.0, 0.0),), n_timepoints=2)
    opera_to_ome_tiff(in_path=measurement_dir / "images", out_path=measurement_dir / "OME-TIFF", keep_stacks=True)
    stack = BioImage(measurement_dir / "OME-TIFF" / "r02c02f01.ome.tiff")
    np.testing.assert_array_equal(stack.data, _expected_stack(pixels, 2, 2, 1, n_timepoints=2))


def test_opera_to_ome_tiff_rejects_same_in_and_out_path(opera_dataset):
    measurement_dir, _ = opera_dataset
    with pytest.raises(ValueError, match="same"):
        opera_to_ome_tiff(in_path=measurement_dir / "images", out_path=measurement_dir / "images", mip=True)


@pytest.fixture
def reference_measurement_dir(test_data, tmp_path):
    """A small (cropped) real Opera Phenix export from the reference dataset,
    copied so that converted files are not written into the dataset."""
    measurement = "7ff28d40-ab05-4813-b326-50cfde35c09d"
    src = Path(test_data) / "datasets" / "opera_phenix" / measurement
    return Path(shutil.copytree(src, tmp_path / measurement))


def test_opera_to_ome_tiff_reference_data(reference_measurement_dir):
    images_dir = reference_measurement_dir / "images"
    opera_to_ome_tiff(in_path=images_dir, out_path=reference_measurement_dir / "OME-TIFF", mip=True, keep_stacks=True)

    stack = BioImage(reference_measurement_dir / "OME-TIFF" / "r03c05f02.ome.tiff")
    assert stack.shape == (1, 1, 3, 128, 128)
    assert list(stack.channel_names) == ["DAPI"]
    assert stack.physical_pixel_sizes.X == pytest.approx(0.0941867, rel=1e-5)
    # planes 1, 8 and 15 of a 0.5 um stack
    assert stack.physical_pixel_sizes.Z == pytest.approx(3.5, rel=1e-3)

    raw = np.stack([tifffile.imread(images_dir / "r03c05" / f"r03c05f02p{p:02d}-ch01t01.tiff") for p in (1, 8, 15)])
    np.testing.assert_array_equal(stack.data[0, 0], raw)

    mip = BioImage(reference_measurement_dir / "OME-TIFF-MIP" / "r03c05f02-mip.ome.tiff")
    np.testing.assert_array_equal(mip.data[0, 0, 0], raw.max(axis=0))
