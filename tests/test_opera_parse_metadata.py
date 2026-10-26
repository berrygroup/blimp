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

from .helpers import OPERA_NS_URI, _make_opera_dataset


@pytest.fixture
def metadata_file(tmp_path):
    _make_opera_dataset(tmp_path / "measurement")
    return tmp_path / "measurement" / "images" / "index.xml"


def test_get_ns_reads_namespace_from_root():
    root = ET.Element(f"{{{OPERA_NS_URI}}}EvaluationInputData")
    assert _get_ns(root) == {"harmony": OPERA_NS_URI}


def test_get_plate_metadata_skips_well_list(metadata_file):
    plate_metadata = get_plate_metadata(metadata_file)
    assert len(plate_metadata) == 1
    assert "Well" not in plate_metadata.columns
    assert plate_metadata.loc[0, "Name"] == "synthetic"
    assert plate_metadata.loc[0, "PlateRows"] == "8"


def test_get_channel_metadata_merges_maps_and_excludes_profiles(metadata_file):
    channel_metadata = get_channel_metadata(metadata_file)
    assert list(channel_metadata["ChannelID"]) == ["1", "2"]
    assert list(channel_metadata["ChannelName"]) == ["DAPI", "GFP"]
    assert "FlatfieldProfile" not in channel_metadata.columns


def test_get_image_metadata_one_row_per_file_with_channel_info(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    # 2 wells x 2 fields x 3 planes x 2 channels
    assert len(image_metadata) == 24
    assert set(image_metadata["URL"].str.split("/").str[0]) == {"r03c05", "r03c06"}
    channel_names = image_metadata.groupby("ChannelID")["ChannelName"].unique()
    assert channel_names.loc[1] == ["DAPI"] and channel_names.loc[2] == ["GFP"]
    assert set(image_metadata["MainExcitationWavelength"]) == {500, 600}
    assert set(image_metadata["WellName"]) == {"C05", "C06"}


def test_get_image_metadata_dtypes(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    assert image_metadata["Row"].dtype == np.uint8
    assert image_metadata["PlaneID"].dtype == np.uint16
    assert image_metadata["ImageResolutionX"].dtype == float
    assert image_metadata["StandardFieldID"].dtype == np.uint16


def test_get_image_metadata_standard_field_id_is_top_left_first(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    field_ids = image_metadata[["FieldID", "StandardFieldID"]].drop_duplicates().set_index("FieldID")
    # field 1 lies to the right of field 2
    assert field_ids.loc[1, "StandardFieldID"] == 2
    assert field_ids.loc[2, "StandardFieldID"] == 1


def test_get_image_metadata_abs_time_converted_to_utc(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    assert image_metadata["AbsTime"].dt.tz is None
    assert image_metadata["AbsTime"].min() == pd.Timestamp("2026-10-21T03:30:01.5")


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
