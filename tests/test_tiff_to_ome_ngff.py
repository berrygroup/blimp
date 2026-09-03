"""Tests for tiff_to_ome_ngff.py -- assembling a whole-plate OME-NGFF store
from an existing OME-TIFF pipeline (intensity, labels, features).

test_convert_tiff_well_to_ome_ngff_matches_reference exercises the full
convert_tiff_well_to_ome_ngff pipeline end-to-end, against a small real
fixture (4 fields from the example nd2 file's own nd2_to_ome_tiff(mip=True)
output, real cellpose segmentation, 4x4-binned down to a few MB -- see
tests/_data/datasets/tiff_to_ome_ngff_test/), for both "grid" and
"exact" placement. The rest of the tests below cover the pure logic
(metadata-driven layout, manifest discovery/logging) that pipeline is built
from.
"""
from pathlib import Path
import logging

from ngio import open_ome_zarr_container
import numpy as np
import pandas as pd
import pytest

from blimp.ome_ngff import ensure_plate_exists
from blimp.constants import blimp_config
from blimp.preprocessing.tiff_to_ome_ngff import (
    _discover_well_manifest,
    convert_tiff_well_to_ome_ngff,
    get_field_layout_from_tiff_metadata,
)


def _write_metadata_csv(tiff_dir: Path, nd2_stem: str, rows: list) -> None:
    pd.DataFrame(rows).to_csv(tiff_dir / f"{nd2_stem}_metadata.csv", index=False)


def _write_blank_tiff(path: Path) -> None:
    """A minimal real OME-TIFF, just enough for BioImage to read shape/dtype/
    pixel size back -- avoids needing a real microscope file for pure-layout
    tests."""
    from bioio_base.types import PhysicalPixelSizes
    from bioio_ome_tiff.writers import OmeTiffWriter
    import numpy as np

    OmeTiffWriter.save(
        data=np.zeros((1, 2, 1, 16, 16), dtype="uint16"),
        uri=str(path),
        dim_order="TCZYX",
        channel_names=["DAPI", "GFP"],
        physical_pixel_sizes=PhysicalPixelSizes(1.0, 0.5, 0.5),
    )


@pytest.fixture
def two_field_well(tmp_path):
    nd2_stem = "WellC09_Seq0001"
    tiff_dir = tmp_path
    filenames = [f"{nd2_stem}_0001.ome.tiff", f"{nd2_stem}_0002.ome.tiff"]
    for f in filenames:
        _write_blank_tiff(tiff_dir / f)
    _write_metadata_csv(
        tiff_dir,
        nd2_stem,
        rows=[
            {"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filenames[0]},
            {"field_id": 2, "stage_x_abs": 8.0, "stage_y_abs": 0.0, "filename_ome_tiff": filenames[1]},
        ],
    )
    return nd2_stem, tiff_dir, filenames


def test_get_field_layout_from_tiff_metadata_reads_real_tiff_and_csv(two_field_well):
    nd2_stem, tiff_dir, _ = two_field_well
    layout = get_field_layout_from_tiff_metadata(nd2_stem, tiff_dir)
    assert (layout.row, layout.column) == ("C", 9)
    assert layout.field_ids == [1, 2]
    assert layout.tile_shape == (1, 2, 1, 16, 16)
    assert layout.pixel_size_x == pytest.approx(0.5)
    assert layout.pixel_size_y == pytest.approx(0.5)
    # x_direction defaults to "left" (mirrored, see
    # nd2_to_ome_ngff.get_field_layout): increasing stage-x maps to
    # *decreasing* pixel-x, so field 2 (the larger stage_x) lands to the
    # left of field 1, not the right.
    assert layout.offsets == [(0, 16), (0, 0)]
    assert layout.channel_names == ["DAPI", "GFP"]
    # TIFF pipeline never carries channel colors -- always the default cycle
    assert layout.channel_colors == ["FF0000", "00FF00"]
    # No XYPosLoop-style position-name metadata exists for a TIFF-sourced
    # layout, so names are synthesized from field_id in the same
    # "{well}_{field_id:04d}" shape nd2's own position names use -- this is
    # what FOV_ROI_table entries end up named, instead of falling back to
    # _build_fov_roi_table's generic "FOV_i".
    assert layout.position_names == ["C09_0001", "C09_0002"]


@pytest.fixture
def two_field_well_overlapping(tmp_path):
    """Like ``two_field_well``, but field 2's stage position is 1 physical
    unit (2 px) short of a full tile pitch -- enough for grid clustering to
    still put it in its own column (the gap exceeds tile_extent/2), but not
    an exact tile multiple, so "grid" and "exact" placement disagree."""
    nd2_stem = "WellC09_Seq0001"
    tiff_dir = tmp_path
    filenames = [f"{nd2_stem}_0001.ome.tiff", f"{nd2_stem}_0002.ome.tiff"]
    for f in filenames:
        _write_blank_tiff(tiff_dir / f)
    _write_metadata_csv(
        tiff_dir,
        nd2_stem,
        rows=[
            {"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filenames[0]},
            {"field_id": 2, "stage_x_abs": 7.0, "stage_y_abs": 0.0, "filename_ome_tiff": filenames[1]},
        ],
    )
    return nd2_stem, tiff_dir, filenames


def test_get_field_layout_from_tiff_metadata_grid_snaps_to_tile_pitch(two_field_well_overlapping):
    nd2_stem, tiff_dir, _ = two_field_well_overlapping
    layout = get_field_layout_from_tiff_metadata(nd2_stem, tiff_dir, placement="grid")
    # Snapped flush at an exact multiple of the 16 px tile width, discarding
    # the 2 px the raw stage positions would otherwise overlap by.
    assert layout.offsets == [(0, 16), (0, 0)]


def test_get_field_layout_from_tiff_metadata_exact_uses_raw_stage_offset(two_field_well_overlapping):
    nd2_stem, tiff_dir, _ = two_field_well_overlapping
    layout = get_field_layout_from_tiff_metadata(nd2_stem, tiff_dir, placement="exact")
    # Raw offset from stage position (7.0 / 0.5 px per unit = 14 px), not
    # snapped to the 16 px tile pitch -- field 2 overlaps field 1 by 2 px.
    assert layout.offsets == [(0, 14), (0, 0)]


def test_get_field_layout_from_tiff_metadata_raises_for_missing_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        get_field_layout_from_tiff_metadata("NoSuchWell", tmp_path)


def test_get_field_layout_from_tiff_metadata_raises_when_no_field_tiffs_exist(tmp_path):
    nd2_stem = "WellC09_Seq0001"
    _write_metadata_csv(
        tmp_path,
        nd2_stem,
        rows=[{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": "missing.ome.tiff"}],
    )
    with pytest.raises(FileNotFoundError, match="None of the field TIFFs"):
        get_field_layout_from_tiff_metadata(nd2_stem, tmp_path)


def test_discover_well_manifest_flags_missing_label_and_feature_files(two_field_well, tmp_path, caplog):
    nd2_stem, tiff_dir, filenames = two_field_well
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # only field 1's label file exists
    (label_dir / filenames[0]).write_bytes(b"")

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    # neither field's feature CSV exists

    with caplog.at_level(logging.WARNING):
        manifest = _discover_well_manifest(
            nd2_stem, tiff_dir, label_dirs={"Nuclei": label_dir}, feature_csv_dirs={"Nuclei": feature_dir}
        )

    assert manifest["intensity_exists"].tolist() == [True, True]
    assert manifest["label_exists_Nuclei"].tolist() == [True, False]
    assert manifest["feature_exists_Nuclei"].tolist() == [False, False]
    assert "label 'Nuclei' missing for field_id(s) [2]" in caplog.text
    assert "feature CSV 'Nuclei' missing for field_id(s) [1, 2]" in caplog.text


def test_discover_well_manifest_with_no_label_or_feature_dirs(two_field_well):
    nd2_stem, tiff_dir, _ = two_field_well
    manifest = _discover_well_manifest(nd2_stem, tiff_dir)
    assert manifest["intensity_exists"].tolist() == [True, True]
    assert list(manifest.columns) == [
        "field_id",
        "stage_x_abs",
        "stage_y_abs",
        "filename_ome_tiff",
        "intensity_exists",
    ]


@pytest.mark.data
@pytest.mark.parametrize("placement", ["grid", "exact"])
def test_convert_tiff_well_to_ome_ngff_matches_reference(placement, tmp_path, _ensure_test_data):
    """Full pipeline, real data: stitch the small downsampled fixture and
    compare the result pixel-for-pixel against a reference store that was
    built the same way and reviewed manually in napari (both intensity
    mosaic and label placement) before being pinned here."""
    testdata_config = blimp_config.get_data_config("testdata")
    fixture_dir = Path(testdata_config.DATASET_DIR) / "tiff_to_ome_ngff_test"
    nd2_stem = "WellC09_Channel647,488,561,405_Seq0006"

    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    convert_tiff_well_to_ome_ngff(
        nd2_stem=nd2_stem,
        tiff_dir=fixture_dir / "OME-TIFF-MIP",
        plate_path=plate_path,
        label_dirs={"Nuclei": fixture_dir / "SEGMENTATION"},
        placement=placement,
    )

    actual = open_ome_zarr_container(str(plate_path / "C" / "09" / "mip"))
    expected = open_ome_zarr_container(
        str(Path(testdata_config.RESOURCES_DIR) / f"tiff_to_ome_ngff_{placement}.zarr" / "C" / "09" / "mip")
    )

    np.testing.assert_array_equal(actual.get_image().get_as_numpy(), expected.get_image().get_as_numpy())
    np.testing.assert_array_equal(
        actual.get_label("Nuclei").get_as_numpy(), expected.get_label("Nuclei").get_as_numpy()
    )

    actual_pixel_size = actual.get_image().pixel_size
    expected_pixel_size = expected.get_image().pixel_size
    actual_rois = {
        r.name: r.to_slicing_dict(pixel_size=actual_pixel_size) for r in actual.get_table("FOV_ROI_table").rois()
    }
    expected_rois = {
        r.name: r.to_slicing_dict(pixel_size=expected_pixel_size) for r in expected.get_table("FOV_ROI_table").rois()
    }
    assert actual_rois == expected_rois
