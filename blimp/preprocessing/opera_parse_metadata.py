"""Extract and parse metadata from Perkin-Elmer (Revvity) Opera Phenix
metadata files."""
from typing import Union
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from blimp.preprocessing.operetta_parse_metadata import (
    _remove_ns,
    _xml_to_df,
    _to_well_name,
    load_image_metadata,
)

logger = logging.getLogger(__name__)

__all__ = ["get_plate_metadata", "get_image_metadata", "get_channel_metadata", "load_image_metadata"]

image_metadata_dtypes = {
    "id": str,
    "State": str,
    "URL": str,
    "Row": np.uint8,
    "Col": np.uint8,
    "FieldID": np.uint16,
    "PlaneID": np.uint16,
    "TimepointID": np.int64,
    "SequenceID": np.uint16,
    "GroupID": np.uint16,
    "ChannelID": np.uint8,
    "FlimID": np.uint8,
    "ChannelName": str,
    "ImageType": str,
    "AcquisitionType": str,
    "IlluminationType": str,
    "ChannelType": str,
    "ImageResolutionX": float,
    "ImageResolutionY": float,
    "ImageSizeX": np.uint16,
    "ImageSizeY": np.uint16,
    "BinningX": np.uint8,
    "BinningY": np.uint8,
    "MaxIntensity": int,
    "CameraType": str,
    "PositionX": float,
    "PositionY": float,
    "PositionZ": float,
    "AbsPositionZ": float,
    "MeasurementTimeOffset": float,
    "MainExcitationWavelength": np.uint16,
    "MainEmissionWavelength": np.uint16,
    "ObjectiveMagnification": np.uint8,
    "ObjectiveNA": float,
    "ExposureTime": float,
    "ExcitationPower": float,
    "OrientationMatrix": str,
    "CropArea": str,
    "StandardFieldID": np.uint16,
}

# per-channel map entries that are large encoded blobs rather than
# image-level metadata
_EXCLUDED_CHANNEL_FIELDS = ["FlatfieldProfile", "SkewcropParameters"]


def _get_ns(metadata_xml: ET.Element) -> dict:
    """Get the Harmony namespace dictionary from the root of a metadata file.

    The namespace URI changes between Harmony versions (e.g.
    ``43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV8``), so it is read
    from the file rather than hard-coded.

    Parameters
    ----------
    metadata_xml
        root element of the xml metadata file

    Returns
    -------
    dict
        namespace dictionary with key "harmony"
    """
    return {"harmony": metadata_xml.tag.split("}")[0].lstrip("{")}


def _write_metadata(metadata: pd.DataFrame, out_file: Union[str, Path, None]) -> None:
    if out_file is not None:
        logger.info(f"Save metadata to file: {str(out_file)}")
        if Path(out_file).suffix == ".csv":
            metadata.to_csv(out_file, index=False)
        elif Path(out_file).suffix == ".pkl":
            metadata.to_pickle(out_file)


def get_plate_metadata(metadata_file: Union[str, Path], out_file: Union[str, Path, None] = None) -> pd.DataFrame:
    """Extracts plate metadata from the opera xml file.

    Parameters
    ----------
    metadata_file
        path to the xml metadata file (``images/index.xml``)
    out_file
        enter a file path if this dataframe should be written to file
        (possible extensions are .csv or .pkl)

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = # plates, n_cols = # xml fields)
    """
    metadata_xml = ET.parse(metadata_file).getroot()
    ns = _get_ns(metadata_xml)

    plates_xml = metadata_xml.find("harmony:Plates", namespaces=ns)
    if isinstance(plates_xml, ET.Element):
        plates_xml = plates_xml.findall("harmony:Plate", namespaces=ns)
    else:
        raise ValueError("Opera plate metadata XML not parsed correctly")

    # plates list their wells as repeated empty <Well id=""/> elements,
    # so keep only the scalar fields
    plate_metadata = pd.DataFrame(
        [
            {_remove_ns(field.tag): field.text for field in plate_xml if _remove_ns(field.tag) != "Well"}
            for plate_xml in plates_xml
        ]
    )

    _write_metadata(plate_metadata, out_file)
    return plate_metadata


def get_channel_metadata(metadata_file: Union[str, Path]) -> pd.DataFrame:
    """Extracts per-channel metadata from the opera xml file.

    Unlike the Operetta, the Opera stores channel-specific acquisition
    settings (channel name, resolution, wavelengths, exposure etc.) once
    per channel in the "Maps" section rather than in every image entry.

    Parameters
    ----------
    metadata_file
        path to the xml metadata file (``images/index.xml``)

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = # channels, n_cols = # xml fields)
    """
    metadata_xml = ET.parse(metadata_file).getroot()
    ns = _get_ns(metadata_xml)

    maps_xml = metadata_xml.find("harmony:Maps", namespaces=ns)
    if not isinstance(maps_xml, ET.Element):
        raise ValueError("Opera channel metadata XML not parsed correctly")

    # each <Map> holds one type of information, with one <Entry> per channel
    channel_metadata: dict = {}
    for entry in maps_xml.iterfind("harmony:Map/harmony:Entry", namespaces=ns):
        channel_id = entry.get("ChannelID")
        if channel_id is None:
            continue
        fields = channel_metadata.setdefault(channel_id, {"ChannelID": channel_id})
        for field in entry:
            tag = _remove_ns(field.tag)
            if tag not in _EXCLUDED_CHANNEL_FIELDS:
                fields[tag] = field.text

    if len(channel_metadata) == 0:
        raise ValueError("Opera channel metadata XML not parsed correctly")

    return pd.DataFrame(list(channel_metadata.values()))


def get_image_metadata(metadata_file: Union[str, Path], out_file: Union[str, Path, None] = None) -> pd.DataFrame:
    """Extracts image metadata from the opera xml file.

    Parameters
    ----------
    metadata_file
        path to the xml metadata file (``images/index.xml``)
    out_file
        enter a file path if this dataframe should be written to file
        (possible extensions are .csv or .pkl)

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = # image files, n_cols = # xml fields)
    """
    metadata_xml = ET.parse(metadata_file).getroot()
    ns = _get_ns(metadata_xml)

    images_xml = metadata_xml.find("harmony:Images", namespaces=ns)
    if isinstance(images_xml, ET.Element):
        images_xml = images_xml.findall("harmony:Image", namespaces=ns)
    else:
        raise ValueError("Opera image metadata XML not parsed correctly")

    # convert to dataframe and add channel information
    image_metadata = _xml_to_df(images_xml, "harmony", ns)
    channel_metadata = get_channel_metadata(metadata_file)
    image_metadata = image_metadata.merge(right=channel_metadata, on="ChannelID", how="left")

    # add field indices using standard order (top-left to bottom right, incrementing columns first)
    image_metadata["XCoordinate"] = (image_metadata["PositionX"].astype("float") * 1e9).astype("int")
    image_metadata["YCoordinate"] = (image_metadata["PositionY"].astype("float") * 1e9).astype("int")
    image_metadata["XYCoordinates"] = image_metadata[["XCoordinate", "YCoordinate"]].apply(tuple, axis=1)

    # Number fields from top-left to bottom-right (increase x first)
    unique_int_coords_sorted = sorted(list(set(image_metadata["XYCoordinates"])), key=lambda k: [-k[1], k[0]])
    coord_index = dict(
        zip(
            unique_int_coords_sorted,
            ["%0d" % i for i in range(1, len(unique_int_coords_sorted) + 1)],
        )
    )

    # keep this as StandardFieldID
    image_metadata["StandardFieldID"] = image_metadata["XYCoordinates"].map(coord_index)
    image_metadata = image_metadata.drop(columns=["XYCoordinates", "XCoordinate", "YCoordinate"])

    image_metadata = image_metadata.astype(
        {key: value for key, value in image_metadata_dtypes.items() if key in image_metadata.columns}
    )

    # AbsTime includes a UTC offset, store as (timezone-naive) UTC
    image_metadata["AbsTime"] = pd.to_datetime(image_metadata["AbsTime"], utc=True, format="ISO8601").dt.tz_localize(
        None
    )

    # add a "WellName" identifier
    image_metadata["WellName"] = image_metadata[["Row", "Col"]].apply(lambda x: _to_well_name(x.Row, x.Col), axis=1)

    _write_metadata(image_metadata, out_file)
    return image_metadata
