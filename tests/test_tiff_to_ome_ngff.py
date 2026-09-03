"""Tests for tiff_to_ome_ngff.py -- assembling a whole-plate OME-NGFF store
from an existing OME-TIFF pipeline (intensity, labels, features).

No fixture exercises the full convert_tiff_well_to_ome_ngff pipeline
end-to-end -- it needs real segmentation output. The pipeline was verified
manually instead, against the real example nd2 file's own
nd2_to_ome_tiff(mip=True) output: a 100% exact pixel match against a fresh
nd2_to_ome_ngff conversion of the same file, correct global object IDs for
two seeded fields, and correctly blank/row-free handling for two
deliberately-omitted fields. The tests below cover the pure logic
(metadata-driven layout, manifest discovery/logging) that pipeline is built
from.
"""
from pathlib import Path
import logging

import pandas as pd
import pytest

from blimp.preprocessing.tiff_to_ome_ngff import (
    _discover_well_manifest,
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
