"""Tests for nd2_to_ome_ngff.py -- the nd2-specific glue around the shared
blimp.ome_ngff writer core (see tests/test_ome_ngff.py for the generic
logic: grid clustering, ROI tables, NGFF metadata, plate/well registration).

The full convert_individual_nd2_to_ome_ngff pipeline is untested here against
a real reference .nd2 file (none of the existing reference files are
multi-position well acquisitions); it was instead verified manually against
a real multi-position well file -- see PR #9.
"""
from pathlib import Path

import pytest

from blimp.constants import blimp_config
from blimp.preprocessing.nd2_to_ome_ngff import get_field_layout


@pytest.mark.data
def test_get_field_layout_raises_for_single_position_file_without_well_name(_ensure_test_data):
    """Real single-position .nd2 fixtures have no XYPosLoop and no 'WellXX'
    filename, so neither well-identity signal is available -- this should
    fail clearly rather than silently guessing a well."""
    testdata_config = blimp_config.get_data_config("testdata")
    nd2_path = Path(testdata_config.DATASET_DIR) / "illumination_correction" / "221103_brightfield_488_568_647_1.nd2"
    if not nd2_path.exists():
        pytest.skip(f"Reference file not found: {nd2_path}")

    with pytest.raises(ValueError, match="Could not determine a well identifier"):
        get_field_layout(nd2_path)
