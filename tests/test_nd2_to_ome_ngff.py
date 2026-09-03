"""Tests for nd2_to_ome_ngff.py -- the nd2-specific glue around the shared
blimp.ome_ngff writer core (see tests/test_ome_ngff_*.py for the generic
logic: grid clustering, ROI tables, NGFF metadata, plate/well registration).

No fixture exercises the full convert_individual_nd2_to_ome_ngff pipeline
(well registration, tile placement, pyramid generation) end-to-end: the
existing reference .nd2 files are single-position with no well-plate
metadata, and there is no tool to produce a smaller real one -- the `nd2`
package (the most complete open reader for this proprietary format) has no
writer for the native container, only write_ome_zarr()/write_tiff(). The
pipeline was verified manually instead, against a real multi-position well
file: correct stitching (both "grid" and "continuous" placement -- see
get_field_layout), correct NGFF 0.5 metadata, and (for "continuous"
placement specifically) a 99.84% exact pixel match against the source data,
with differences confined to the expected tile-overlap zones from
last-write-wins placement.
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
