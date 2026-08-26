import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from blimp.preprocessing.operetta_parse_metadata import (
    _remove_ns,
    _xml_to_df,
    _to_well_name,
    get_image_metadata,
    get_plate_metadata,
)

NS_URI = "http://www.perkinelmer.com/PEHH/HarmonyV5"
NS = {"harmony": NS_URI}


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


def _plate_xml(n_plates=2):
    root = ET.Element(f"{{{NS_URI}}}Plates")
    for i in range(n_plates):
        plate = ET.SubElement(root, f"{{{NS_URI}}}Plate")
        ET.SubElement(plate, f"{{{NS_URI}}}Name").text = f"plate-{i}"
        ET.SubElement(plate, f"{{{NS_URI}}}PlateTypeName").text = "384well"
        ET.SubElement(plate, f"{{{NS_URI}}}PlateRows").text = "16"
        ET.SubElement(plate, f"{{{NS_URI}}}PlateColumns").text = "24"
    return root


def test_xml_to_df_returns_one_row_per_element():
    root = _plate_xml(n_plates=3)
    plates = root.findall("harmony:Plate", namespaces=NS)
    df = _xml_to_df(plates, "harmony", NS)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert {"Name", "PlateTypeName", "PlateRows", "PlateColumns"}.issubset(df.columns)
    assert list(df["Name"]) == ["plate-0", "plate-1", "plate-2"]


def test_xml_to_df_handles_single_element():
    root = _plate_xml(n_plates=1)
    df = _xml_to_df(root.findall("harmony:Plate", namespaces=NS), "harmony", NS)
    assert len(df) == 1


def _write(tmp_path, text, name="Index.idx.xml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_get_plate_metadata_raises_on_malformed_xml(tmp_path):
    """Regression: this previously called os._exit(1), killing the interpreter
    (and any pytest run or batch job) instead of raising."""
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
