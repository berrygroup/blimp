"""Tests for operetta_parse_metadata.py, on the metadata file of a real
Operetta CLS export from the reference dataset (cropped to one well, two
fields, three planes and three channels)."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from blimp.preprocessing.operetta_parse_metadata import (
    _remove_ns,
    _xml_to_df,
    _to_well_name,
    get_image_metadata,
    get_plate_metadata,
    load_image_metadata,
)

NS_URI = "http://www.perkinelmer.com/PEHH/HarmonyV5"
NS = {"harmony": NS_URI}


@pytest.fixture
def metadata_file(test_data):
    measurement = "20240905_Cycle1_4i_experiment__2024-09-05T20_03_34-Measurement_1"
    return Path(test_data) / "datasets" / "operetta" / measurement / "Images" / "Index.idx.xml"


def test_remove_ns_strips_namespace():
    assert _remove_ns("{" + NS_URI + "}Plates") == "Plates"


def test_remove_ns_keeps_remainder_with_braces():
    assert _remove_ns("{ns}Name") == "Name"


@pytest.mark.parametrize(
    "row,column,expected",
    [(1, 1, "A01"), (1, 13, "A13"), (2, 3, "B03"), (8, 12, "H12"), (16, 24, "P24")],
)
def test_to_well_name(row, column, expected):
    assert _to_well_name(row, column) == expected


def test_to_well_name_zero_pads_single_digit_columns():
    assert _to_well_name(3, 4) == "C04"


def test_xml_to_df_returns_one_row_per_element(metadata_file):
    images = ET.parse(metadata_file).getroot().find("harmony:Images", namespaces=NS).findall("harmony:Image", NS)
    df = _xml_to_df(images, "harmony", NS)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 18
    assert {"id", "URL", "ChannelName", "AbsTime"}.issubset(df.columns)
    assert df.loc[0, "URL"] == "r02c02f01p01-ch1sk1fk1fl1.tiff"


def test_xml_to_df_handles_single_element(metadata_file):
    plates = ET.parse(metadata_file).getroot().find("harmony:Plates", namespaces=NS).findall("harmony:Plate", NS)
    df = _xml_to_df(plates, "harmony", NS)
    assert len(df) == 1
    assert df.loc[0, "Name"] == "20240905_Cycle1_4i_experiment"


def _write(tmp_path, text, name="Index.idx.xml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_get_plate_metadata_raises_on_malformed_xml(tmp_path):
    bad = _write(tmp_path, f'<?xml version="1.0"?><Root xmlns="{NS_URI}"><NotPlates/></Root>')
    with pytest.raises((ValueError, AttributeError, TypeError)):
        get_plate_metadata(bad)


def test_get_image_metadata_raises_on_malformed_xml(tmp_path):
    bad = _write(tmp_path, f'<?xml version="1.0"?><Root xmlns="{NS_URI}"><NotImages/></Root>')
    with pytest.raises((ValueError, AttributeError, TypeError)):
        get_image_metadata(bad)


def test_get_plate_metadata_raises_on_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        get_plate_metadata(tmp_path / "does_not_exist.xml")


def test_get_plate_metadata_parses_valid_xml(metadata_file):
    plate_metadata = get_plate_metadata(metadata_file)
    assert len(plate_metadata) == 1
    assert plate_metadata.loc[0, "Name"] == "20240905_Cycle1_4i_experiment"
    assert plate_metadata.loc[0, "PlateTypeName"] == "96 Greiner uClear"
    assert plate_metadata.loc[0, "PlateRows"] == "8"
    assert plate_metadata.loc[0, "PlateColumns"] == "12"


def test_get_image_metadata_parses_valid_xml(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    # 1 well x 2 fields x 3 planes x 3 channels
    assert len(image_metadata) == 18
    channel_names = image_metadata[["ChannelID", "ChannelName"]].drop_duplicates().set_index("ChannelID")
    assert channel_names["ChannelName"].to_dict() == {1: "Alexa 568", 2: "Alexa 647", 3: "DAPI"}
    assert (image_metadata["WellName"] == "B02").all()
    assert (image_metadata["TimepointID"] == 0).all()
    assert sorted(image_metadata["PlaneID"].unique()) == [1, 8, 14]
    assert image_metadata.loc[0, "ImageResolutionX"] == pytest.approx(1.4949402023919043e-07)


def test_get_image_metadata_dtypes(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    assert image_metadata["Row"].dtype == np.uint8
    assert image_metadata["PlaneID"].dtype == np.uint16
    assert image_metadata["TimepointID"].dtype == np.int64
    assert image_metadata["ImageResolutionX"].dtype == float
    assert image_metadata["StandardFieldID"].dtype == np.uint16


def test_get_image_metadata_keeps_field_id_and_adds_standard_field_id(metadata_file):
    image_metadata = get_image_metadata(metadata_file)
    field_ids = image_metadata[["FieldID", "StandardFieldID"]].drop_duplicates().set_index("FieldID")
    # FieldID is as recorded by the instrument; StandardFieldID numbers fields
    # top-left first, and field 4 lies above field 1
    assert sorted(field_ids.index) == [1, 4]
    assert field_ids.loc[4, "StandardFieldID"] == 1
    assert field_ids.loc[1, "StandardFieldID"] == 2


def test_get_image_metadata_abs_time_converted_to_utc(metadata_file):
    image_metadata = get_image_metadata(metadata_file).set_index("URL")
    assert image_metadata["AbsTime"].dt.tz is None
    # 2024-09-05T20:03:37.457+10:00
    assert image_metadata["AbsTime"].min() == pd.Timestamp("2024-09-05T10:03:37.457")
    # recorded on a whole second, with no fractional part: 2024-09-05T20:04:57+10:00
    assert image_metadata.loc["r02c02f04p08-ch1sk1fk1fl1.tiff", "AbsTime"] == pd.Timestamp("2024-09-05T10:04:57")


@pytest.mark.parametrize("suffix", [".csv", ".pkl"])
def test_get_image_metadata_writes_file(metadata_file, tmp_path, suffix):
    out_file = tmp_path / f"image_metadata{suffix}"
    image_metadata = get_image_metadata(metadata_file, out_file)
    assert len(load_image_metadata(out_file)) == len(image_metadata)
