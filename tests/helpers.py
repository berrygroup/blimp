from typing import List, Union
from pathlib import Path
import os
import logging
import zipfile

from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.constants import blimp_config

logger = logging.getLogger(__name__)


# -- FIXTURES ---
@pytest.fixture()
def _ensure_test_data():
    from blimp.constants import SCRIPTS_DIR

    if os.path.isdir(os.path.join(SCRIPTS_DIR, "tests/_experiments")) and os.path.isdir(
        os.path.join(SCRIPTS_DIR, "tests/_data")
    ):
        return
    with zipfile.ZipFile(os.path.join(SCRIPTS_DIR, "tests/_test_data.zip"), "r") as zip_ref:
        zip_ref.extractall(os.path.join(SCRIPTS_DIR, "tests/."))


def _load_test_data(dataset: str) -> Union[List[AICSImage], None]:
    testdata_config = blimp_config.get_data_config("testdata")
    if dataset == "operetta_cls_multiplex":
        dataset_path = Path(testdata_config.DATASET_DIR) / "operetta_cls_multiplex"
        logger.info(f"Loading multiplexed images from {dataset_path}")
        cyc01 = AICSImage(dataset_path / "cycle_01" / "r05c03f15-fk1fl1-mip.ome.tiff")
        cyc02 = AICSImage(dataset_path / "cycle_02" / "r05c03f15-fk1fl1-mip.ome.tiff")
        return [cyc01, cyc02]
    elif dataset == "registration_tests":
        dataset_path = Path(testdata_config.DATASET_DIR) / "registration_tests"
        logger.info(f"Loading registration test images from {dataset_path}")
        dapi01 = AICSImage(np.load(dataset_path / "dapi01.npy"))
        dapi01_translated = AICSImage(np.load(dataset_path / "dapi01_translated.npy"))
        dapi01_rotated = AICSImage(np.load(dataset_path / "dapi01_rotated.npy"))
        dapi01_rotated_registered = AICSImage(np.load(dataset_path / "dapi01_rotated_registered.npy"))
        return [dapi01, dapi01_translated, dapi01_rotated, dapi01_rotated_registered]
    elif dataset == "illumination_correction":
        dataset_path = Path(testdata_config.DATASET_DIR) / "illumination_correction"
        logger.info(f"Loading registration test images from {dataset_path}")
        bf01 = AICSImage(dataset_path / "221103_brightfield_488_568_647_1.nd2")
        bf02 = AICSImage(dataset_path / "221103_brightfield_488_568_647_2.nd2")
        return [bf01, bf02]
    elif dataset == "synthetic_2D":
        dataset_path = Path(testdata_config.DATASET_DIR) / "synthetic_images"
        logger.info(f"Loading registration test images from {dataset_path}")
        syn_intensity01 = AICSImage(dataset_path / "synthetic_intensity_image_TYX.tiff")
        syn_label02 = AICSImage(dataset_path / "synthetic_label_image_TYX.tiff")
        return [syn_intensity01, syn_label02]
    elif dataset == "synthetic_3D":
        dataset_path = Path(testdata_config.DATASET_DIR) / "synthetic_images"
        logger.info(f"Loading registration test images from {dataset_path}")
        syn_intensity01 = AICSImage(dataset_path / "synthetic_intensity_image_TZYX.tiff")
        syn_label02 = AICSImage(dataset_path / "synthetic_label_image_TZYX.tiff")
        return [syn_intensity01, syn_label02]

    return None
