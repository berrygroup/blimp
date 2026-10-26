from typing import Dict, List, Tuple, Union
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

from bioio import BioImage
import numpy as np
import pytest
import tifffile

from blimp.constants import blimp_config

logger = logging.getLogger(__name__)


# -- FIXTURES ---


def _load_test_data(dataset: str) -> Union[List[BioImage], None]:
    """Load a named reference dataset.

    Skips the calling test if the reference dataset has not been downloaded, so
    that the suite remains runnable offline.
    """
    testdata_config = blimp_config.get_data_config("testdata")
    if testdata_config.DATASET_DIR is None or not Path(testdata_config.DATASET_DIR).is_dir():
        pytest.skip(f"Reference dataset not available (needed for '{dataset}'). Use --download-test-data.")
    if dataset == "operetta_cls_multiplex":
        dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"
        logger.info(f"Loading multiplexed images from {dataset_path}")
        cyc01 = BioImage(dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff")
        cyc02 = BioImage(dataset_path / "cycle_02" / "r05c03f15-fk1fl1-mip.ome.tiff")
        return [cyc01, cyc02]
    elif dataset == "registration_tests":
        dataset_path = Path(testdata_config.DATASET_DIR) / "registration_tests"
        logger.info(f"Loading registration test images from {dataset_path}")
        dapi01 = BioImage(np.load(dataset_path / "dapi01.npy"))
        dapi01_translated = BioImage(np.load(dataset_path / "dapi01_translated.npy"))
        dapi01_rotated = BioImage(np.load(dataset_path / "dapi01_rotated.npy"))
        dapi01_rotated_registered = BioImage(np.load(dataset_path / "dapi01_rotated_registered.npy"))
        return [dapi01, dapi01_translated, dapi01_rotated, dapi01_rotated_registered]
    elif dataset == "illumination_correction":
        dataset_path = Path(testdata_config.DATASET_DIR) / "illumination_correction"
        logger.info(f"Loading registration test images from {dataset_path}")
        bf01 = BioImage(dataset_path / "221103_brightfield_488_568_647_1.nd2")
        bf02 = BioImage(dataset_path / "221103_brightfield_488_568_647_2.nd2")
        return [bf01, bf02]
    elif dataset == "synthetic_2D":
        dataset_path = Path(testdata_config.DATASET_DIR) / "synthetic_images"
        logger.info(f"Loading quantify (2D) test images from {dataset_path}")
        syn_intensity01 = BioImage(dataset_path / "synthetic_intensity_image_TYX.tiff")
        syn_label02 = BioImage(dataset_path / "synthetic_label_image_TYX.tiff")
        return [syn_intensity01, syn_label02]
    elif dataset == "synthetic_3D":
        dataset_path = Path(testdata_config.DATASET_DIR) / "synthetic_images"
        logger.info(f"Loading quantify (3D) test images from {dataset_path}")
        syn_intensity01 = BioImage(dataset_path / "synthetic_intensity_image_TZYX.tiff")
        syn_label02 = BioImage(dataset_path / "synthetic_label_image_TZYX.tiff")
        return [syn_intensity01, syn_label02]

    return None


# -- SYNTHETIC OPERA DATASET ---

OPERA_NS_URI = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV8"


def _make_opera_dataset(
    measurement_dir: Union[str, Path],
    wells: Tuple[Tuple[int, int], ...] = ((3, 5), (3, 6)),
    field_positions: Tuple[Tuple[float, float], ...] = ((1e-4, 0.0), (0.0, 0.0)),
    n_planes: int = 3,
    channel_names: Tuple[str, ...] = ("DAPI", "GFP"),
    n_timepoints: int = 1,
    image_size: int = 16,
) -> Dict[Tuple[int, int, int, int, int, int], np.ndarray]:
    """Write a minimal Opera Phenix export to ``measurement_dir``.

    Mirrors the layout of a real export: ``images/index.xml`` (HarmonyV8
    namespace, per-channel settings in "Maps") plus one subdirectory of
    single-plane TIFFs per well. Image entries are written in reverse plane
    order, so readers must sort by z position rather than rely on file order.
    Field 1 lies to the right of field 2 with the default positions, so
    ``StandardFieldID`` (top-left first) is the reverse of ``FieldID``.

    Returns
    -------
    dict
        pixel data keyed by (Row, Col, FieldID, PlaneID, ChannelID, TimepointID)
    """
    measurement_dir = Path(measurement_dir)
    images_dir = measurement_dir / "images"
    images_dir.mkdir(parents=True)

    def sub(parent, tag, text=None, **attrib):
        element = ET.SubElement(parent, f"{{{OPERA_NS_URI}}}{tag}", attrib)
        if text is not None:
            element.text = str(text)
        return element

    root = ET.Element(f"{{{OPERA_NS_URI}}}EvaluationInputData", {"Version": "2"})
    sub(root, "User", "berrylab")
    plate = sub(sub(root, "Plates"), "Plate")
    sub(plate, "PlateID", "synthetic")
    sub(plate, "MeasurementID", "00000000-0000-0000-0000-000000000000")
    sub(plate, "Name", "synthetic")
    sub(plate, "PlateTypeName", "96 Greiner uClear")
    sub(plate, "PlateRows", 8)
    sub(plate, "PlateColumns", 12)
    for row, col in wells:
        sub(plate, "Well", id=f"{row:02d}{col:02d}")

    maps = sub(root, "Maps")
    flatfield_map = sub(maps, "Map")
    channel_map = sub(maps, "Map")
    for channel_id, channel_name in enumerate(channel_names, start=1):
        sub(sub(flatfield_map, "Entry", ChannelID=str(channel_id)), "FlatfieldProfile", "{Background: {}}")
        entry = sub(channel_map, "Entry", ChannelID=str(channel_id))
        for tag, text in [
            ("ChannelName", channel_name),
            ("ImageType", "Signal"),
            ("AcquisitionType", "NipkowConfocal"),
            ("IlluminationType", "Epifluorescence"),
            ("ChannelType", "Fluorescence"),
            ("ImageResolutionX", "1E-07"),
            ("ImageResolutionY", "1E-07"),
            ("ImageSizeX", image_size),
            ("ImageSizeY", image_size),
            ("BinningX", 1),
            ("BinningY", 1),
            ("MaxIntensity", 65535),
            ("CameraType", "Tucsen"),
            ("MainExcitationWavelength", 400 + 100 * channel_id),
            ("MainEmissionWavelength", 450 + 100 * channel_id),
            ("ObjectiveMagnification", 63),
            ("ObjectiveNA", 1.15),
            ("ExposureTime", 0.08),
            ("ExcitationPower", 50),
            ("OrientationMatrix", "[[0.993265,0,0,-98.8],[0,-0.993265,0,9.3],[0,0,1.00,-2.151]]"),
            ("CropArea", f"[[0,0],[{image_size},{image_size}],[{image_size},{image_size}]]"),
        ]:
            sub(entry, tag, text)

    rng = np.random.default_rng(0)
    pixels = {}
    images = sub(root, "Images")
    for row, col in wells:
        well_dir = images_dir / f"r{row:02d}c{col:02d}"
        well_dir.mkdir()
        for field_id, (x, y) in enumerate(field_positions, start=1):
            for timepoint_id in range(1, n_timepoints + 1):
                for channel_id in range(1, len(channel_names) + 1):
                    for plane_id in reversed(range(1, n_planes + 1)):
                        filename = f"r{row:02d}c{col:02d}f{field_id:02d}p{plane_id:02d}-ch{channel_id:02d}t{timepoint_id:02d}.tiff"
                        data = rng.integers(0, 4096, size=(image_size, image_size), dtype=np.uint16)
                        tifffile.imwrite(well_dir / filename, data)
                        pixels[(row, col, field_id, plane_id, channel_id, timepoint_id)] = data

                        image = sub(images, "Image", Version="1")
                        for tag, text in [
                            ("id", f"{row:02d}{col:02d}K{timepoint_id}F{field_id}P{plane_id}R{channel_id}"),
                            ("State", "Ok"),
                            ("URL", f"{well_dir.name}/{filename}"),
                            ("Row", row),
                            ("Col", col),
                            ("FieldID", field_id),
                            ("PlaneID", plane_id),
                            ("TimepointID", timepoint_id),
                            ("SequenceID", 1),
                            ("GroupID", 1),
                            ("ChannelID", channel_id),
                            ("FlimID", 1),
                            ("PositionX", x),
                            ("PositionY", y),
                            ("PositionZ", (plane_id - 1) * 5e-7),
                            ("AbsPositionZ", 0.1 + (plane_id - 1) * 5e-7),
                            ("MeasurementTimeOffset", 0),
                            ("AbsTime", f"2026-10-21T14:{29 + timepoint_id:02d}:{field_id:02d}.5+11:00"),
                        ]:
                            sub(image, tag, text)

    ET.ElementTree(root).write(images_dir / "index.xml", encoding="utf-8", xml_declaration=True)
    return pixels
