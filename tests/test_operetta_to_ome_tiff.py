"""Tests for operetta_to_ome_tiff.py -- conversion of a synthetic Operetta
export (see tests/helpers.py) to OME-TIFF."""

from bioio import BioImage
import numpy as np
import pandas as pd
import pytest

from blimp.preprocessing.operetta_to_ome_tiff import (
    _get_zyx_resolution,
    _remove_TCZ_filename,
    operetta_to_ome_tiff,
    _aggregate_TCZ_metadata,
    operetta_regex_filename_pattern,
)
from blimp.preprocessing.operetta_parse_metadata import get_image_metadata

from .helpers import _make_operetta_dataset


@pytest.mark.parametrize(
    "filename,mip,expected",
    [
        ("r03c05f01p01-ch1sk1fk1fl1.tiff", False, "r03c05f01-fk1fl1.ome.tiff"),
        ("r03c05f12p15-ch2sk3fk1fl1.tiff", True, "r03c05f12-fk1fl1-mip.ome.tiff"),
        ("Index.idx.xml", False, None),
    ],
)
def test_remove_TCZ_filename(filename, mip, expected):
    assert _remove_TCZ_filename(operetta_regex_filename_pattern, filename, mip) == expected


@pytest.fixture
def operetta_dataset(tmp_path):
    experiment_dir = tmp_path / "experiment"
    pixels = _make_operetta_dataset(experiment_dir)
    return experiment_dir, pixels


def test_aggregate_TCZ_metadata_counts_planes_and_channels(operetta_dataset):
    experiment_dir, _ = operetta_dataset
    image_metadata = get_image_metadata(experiment_dir / "Images" / "Index.idx.xml")
    aggregated = _aggregate_TCZ_metadata(image_metadata)
    assert len(aggregated) == 4
    assert (aggregated["NPlanes"] == 3).all()
    assert (aggregated["NChannels"] == 2).all()


def test_get_zyx_resolution_in_micrometres(operetta_dataset):
    experiment_dir, _ = operetta_dataset
    image_metadata = get_image_metadata(experiment_dir / "Images" / "Index.idx.xml")
    voxel_dimensions = _get_zyx_resolution(image_metadata, 3, 5, 1, 1, 0)
    assert voxel_dimensions.X == pytest.approx(0.1)
    assert voxel_dimensions.Y == pytest.approx(0.1)
    assert voxel_dimensions.Z == pytest.approx(0.5)


def test_get_zyx_resolution_single_plane_has_no_z(tmp_path):
    _make_operetta_dataset(tmp_path / "experiment", n_planes=1)
    image_metadata = get_image_metadata(tmp_path / "experiment" / "Images" / "Index.idx.xml")
    assert _get_zyx_resolution(image_metadata, 3, 5, 1, 1, 0).Z is None


def test_operetta_to_ome_tiff_writes_stacks_and_mips(operetta_dataset):
    experiment_dir, pixels = operetta_dataset
    out_path = experiment_dir / "OME-TIFF"
    operetta_to_ome_tiff(
        in_path=experiment_dir / "Images",
        out_path=out_path,
        metadata_file="Index.idx.xml",
        mip=True,
        keep_stacks=True,
    )

    assert sorted(p.name for p in out_path.glob("*.ome.tiff")) == [
        "r03c05f01-fk1fl1.ome.tiff",
        "r03c05f02-fk1fl1.ome.tiff",
        "r03c06f01-fk1fl1.ome.tiff",
        "r03c06f02-fk1fl1.ome.tiff",
    ]
    expected = np.stack([np.stack([pixels[(3, 6, 2, p, c, 0)] for p in range(1, 4)]) for c in range(1, 3)])[np.newaxis]

    stack = BioImage(out_path / "r03c06f02-fk1fl1.ome.tiff")
    assert stack.shape == (1, 2, 3, 16, 16)
    assert list(stack.channel_names) == ["DAPI", "GFP"]
    assert stack.physical_pixel_sizes.Z == pytest.approx(0.5)
    assert stack.physical_pixel_sizes.X == pytest.approx(0.1)
    np.testing.assert_array_equal(stack.data, expected)

    mip = BioImage(experiment_dir / "OME-TIFF-MIP" / "r03c06f02-fk1fl1-mip.ome.tiff")
    np.testing.assert_array_equal(mip.data, expected.max(axis=2, keepdims=True))

    metadata = pd.read_csv(out_path / "image_metadata.csv")
    assert len(metadata) == 4
    assert (metadata["NPlanes"] == 3).all()
