"""Tests for opera_parse_metadata.py, on the metadata file of a real Opera
Phenix export from the reference dataset (cropped to one well, two fields
and three planes, with one channel)."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from blimp.preprocessing.opera_parse_metadata import (
    _get_ns,
    get_image_metadata,
    get_plate_metadata,
    load_image_metadata,
    get_channel_metadata,
)

OPERA_NS_URI = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV8"


@pytest.fixture
def metadata_file(test_data):
    measurement = "7ff28d40-ab05-4813-b326-50cfde35c09d"
    return Path(test_data) / "datasets" / "opera_phenix" / measurement / "images" / "index.xml"


def test_get_ns_reads_namespace_from_root(metadata_file):
    assert _get_ns(ET.parse(metadata_file).getroot()) == {"harmony": OPERA_NS_URI}


def test_get_plate_metadata_skips_well_list(metadata_file):
    plate_metadata = get_plate_metadata(metadata_file)
    assert len(plate_metadata) == 1
    assert "Well" not in plate_metadata.columns
    assert plate_metadata.loc[0, "Name"] == "hipsctest"
    assert plate_metadata.loc[0, "PlateTypeName"] == "96 Greiner uClear"
    assert plate_metadata.loc[0, "PlateRows"] == "8"
    assert plate_metadata.loc[0, "PlateColumns"] == "12"


def test_get_channel_metadata_merges_maps_and_excludes_profiles(metadata_file):
    channel_metadata = get_channel_metadata(metadata_file)
    assert len(channel_metadata) == 1
    channel = channel_metadata.iloc[0]
    assert channel["ChannelID"] == "1"
    assert channel["ChannelName"] == "DAPI"
    assert channel["MainExcitationWavelength"] == "405"
    assert channel["MainEmissionWavelength"] == "456"
    assert channel["ObjectiveMagnification"] == "63"
    # flatfield and skew-crop maps hold encoded blobs, not image metadata
    assert "FlatfieldProfile" not in channel_metadata.columns
    assert "SkewcropParameters" not in channel_metadata.columns


def test_get_image_metadata_one_row_per_file_with_channel_info(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    # 1 well x 2 fields x 3 planes x 1 channel
    assert len(image_metadata) == 6
    assert image_metadata.loc[0, "URL"] == "r03c05/r03c05f01p01-ch01t01.tiff"
    assert (image_metadata["ChannelName"] == "DAPI").all()
    assert (image_metadata["WellName"] == "C05").all()
    assert sorted(image_metadata["PlaneID"].unique()) == [1, 8, 15]
    assert image_metadata.loc[0, "ImageResolutionX"] == pytest.approx(9.41867087221273e-08)


def test_get_image_metadata_dtypes(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    assert image_metadata["Row"].dtype == np.uint8
    assert image_metadata["PlaneID"].dtype == np.uint16
    assert image_metadata["TimepointID"].dtype == np.int64
    assert image_metadata["ImageResolutionX"].dtype == float
    assert image_metadata["StandardFieldID"].dtype == np.uint16


def test_get_image_metadata_standard_field_id_is_top_left_first(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    field_ids = image_metadata[["FieldID", "StandardFieldID"]].drop_duplicates().set_index("FieldID")
    # field 1 (x = 0) lies to the right of field 2 (x = -0.77 mm)
    assert field_ids.loc[1, "StandardFieldID"] == 2
    assert field_ids.loc[2, "StandardFieldID"] == 1


def test_get_image_metadata_abs_time_converted_to_utc(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    assert image_metadata["AbsTime"].dt.tz is None
    # 2026-10-21T14:30:30.6055988+11:00
    assert image_metadata["AbsTime"].min() == pd.Timestamp("2026-10-21T03:30:30.6055988")


@pytest.mark.parametrize("suffix", [".csv", ".pkl"])
def test_get_image_metadata_writes_file(metadata_file, tmp_path, suffix):
    out_file = tmp_path / f"image_metadata{suffix}"
    image_metadata = get_image_metadata(metadata_file, out_file)
    assert len(load_image_metadata(out_file)) == len(image_metadata)


def _write(tmp_path, text, name="index.xml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_get_plate_metadata_raises_on_malformed_xml(tmp_path):
    bad = _write(tmp_path, f'<?xml version="1.0"?><Root xmlns="{OPERA_NS_URI}"><NotPlates/></Root>')
    with pytest.raises(ValueError):
        get_plate_metadata(bad)


def test_get_image_metadata_raises_on_malformed_xml(tmp_path):
    bad = _write(tmp_path, f'<?xml version="1.0"?><Root xmlns="{OPERA_NS_URI}"><NotImages/></Root>')
    with pytest.raises(ValueError):
        get_image_metadata(bad)


def test_get_channel_metadata_raises_without_channel_entries(tmp_path):
    bad = _write(tmp_path, f'<?xml version="1.0"?><Root xmlns="{OPERA_NS_URI}"><Maps><Map/></Maps></Root>')
    with pytest.raises(ValueError):
        get_channel_metadata(bad)


def test_get_plate_metadata_raises_on_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        get_plate_metadata(tmp_path / "does_not_exist.xml")
