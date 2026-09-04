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
    _get_parent_channel_name,
    _is_point_object_channel,
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


def _write_label_tiff(path: Path, channel_arrays: dict, pixel_size: float = 0.5) -> None:
    """A small real multi-channel label TIFF: one channel per (name, 2D
    array) pair, stacked in the given order."""
    from bioio_base.types import PhysicalPixelSizes
    from bioio_ome_tiff.writers import OmeTiffWriter

    channel_names = list(channel_arrays.keys())
    data = np.stack([channel_arrays[name] for name in channel_names])[np.newaxis, :, np.newaxis, :, :]
    OmeTiffWriter.save(
        data=data.astype("uint16"),
        uri=str(path),
        dim_order="TCZYX",
        channel_names=channel_names,
        physical_pixel_sizes=PhysicalPixelSizes(1.0, pixel_size, pixel_size),
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
        manifest = _discover_well_manifest(nd2_stem, tiff_dir, label_dir=label_dir, feature_csv_dir=feature_dir)

    assert manifest["intensity_exists"].tolist() == [True, True]
    assert manifest["label_exists"].tolist() == [True, False]
    assert manifest["feature_exists"].tolist() == [False, False]
    assert "label TIFF missing for field_id(s) [2]" in caplog.text
    assert "feature CSV missing for field_id(s) [1, 2]" in caplog.text


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


def test_get_parent_channel_name_reads_column():
    df = pd.DataFrame({"label": [1, 2], "parent_label_name": ["Nuclei", "Nuclei"]})
    assert _get_parent_channel_name(df) == "Nuclei"


def test_get_parent_channel_name_raises_when_column_absent():
    df = pd.DataFrame({"label": [1, 2]})
    with pytest.raises(ValueError, match="parent_label_name"):
        _get_parent_channel_name(df)


def test_is_point_object_channel_explicit_name_wins():
    # No feature CSV at all -- the explicit name is the only signal.
    assert _is_point_object_channel("Spots", None, None, ["Spots"]) is True
    assert _is_point_object_channel("Nuclei", None, None, ["Spots"]) is False


def test_is_point_object_channel_reads_parent_own_column():
    df = pd.DataFrame({"label": [1], "parent_label_name": ["Nuclei"], "is_point_object": [False]})
    assert _is_point_object_channel("Nuclei", "Nuclei", df, None) is False


def test_is_point_object_channel_reads_child_prefixed_column():
    df = pd.DataFrame(
        {
            "label": [1],
            "parent_label_name": ["Nuclei"],
            "is_point_object": [False],
            "Spots_is_point_object": [True],
        }
    )
    assert _is_point_object_channel("Spots", "Nuclei", df, None) is True


def test_is_point_object_channel_falls_back_to_blob_and_warns(caplog):
    df = pd.DataFrame({"label": [1], "parent_label_name": ["Nuclei"], "is_point_object": [False]})
    with caplog.at_level(logging.WARNING):
        # "Cell" has no f"Cell_is_point_object" column and isn't named explicitly.
        assert _is_point_object_channel("Cell", "Nuclei", df, None) is False
    assert "Could not determine whether 'Cell' is a point-object channel" in caplog.text


def test_convert_tiff_well_to_ome_ngff_writes_every_channel_only_parent_gets_features(tmp_path):
    """Regression guard for the old label_image.get_image_data(..., C=0)
    hardcoding: a genuinely multi-channel label TIFF must have each channel
    read from its own index, not channel 0 broadcast to every label -- and
    only the channel quantify()'s own aggregation was built around gets a
    FeatureTable."""
    nd2_stem = "WellC09_Seq0001"
    tiff_dir = tmp_path / "intensity"
    tiff_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    filename = f"{nd2_stem}_0001.ome.tiff"
    _write_blank_tiff(tiff_dir / filename)

    nuclei_array = np.zeros((16, 16), dtype="uint16")
    nuclei_array[0:8, 0:8] = 1
    nuclei_array[8:16, 8:16] = 2
    cell_array = np.zeros((16, 16), dtype="uint16")
    cell_array[0:8, :] = 5
    cell_array[8:16, :] = 6
    _write_label_tiff(label_dir / filename, {"Nuclei": nuclei_array, "Cell": cell_array})

    pd.DataFrame(
        {
            "label": [1, 2],
            "parent_label_name": ["Nuclei", "Nuclei"],
            "is_point_object": [False, False],
            "Cell_count": [3, 2],
            "Cell_is_point_object": [False, False],
        }
    ).to_csv(feature_dir / f"{Path(filename).stem}.csv", index=False)

    _write_metadata_csv(
        tiff_dir,
        nd2_stem,
        rows=[{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}],
    )

    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    convert_tiff_well_to_ome_ngff(
        nd2_stem=nd2_stem,
        tiff_dir=tiff_dir,
        plate_path=plate_path,
        label_dir=label_dir,
        feature_csv_dir=feature_dir,
    )

    container = open_ome_zarr_container(str(plate_path / "C" / "09" / "mip"))
    assert set(container.list_labels()) == {"Nuclei", "Cell"}

    nuclei_values = set(np.unique(container.get_label("Nuclei").get_as_numpy())) - {0}
    cell_values = set(np.unique(container.get_label("Cell").get_as_numpy())) - {0}
    assert nuclei_values, "Nuclei channel should have real (non-background) values"
    # The old bug always read channel 0 (Nuclei) for every label -- these
    # must be genuinely different data, not the same array read twice.
    assert nuclei_values != cell_values

    assert "Nuclei_features" in container.list_tables()
    assert "Cell_features" not in container.list_tables()


def test_convert_tiff_well_to_ome_ngff_routes_named_channel_to_generic_roi_table(tmp_path):
    nd2_stem = "WellC09_Seq0001"
    tiff_dir = tmp_path / "intensity"
    tiff_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()

    filename = f"{nd2_stem}_0001.ome.tiff"
    _write_blank_tiff(tiff_dir / filename)

    nuclei_array = np.zeros((16, 16), dtype="uint16")
    nuclei_array[0:8, 0:8] = 1
    spots_mask = np.zeros((16, 16), dtype="uint16")
    spots_mask[2, 2] = 1
    spots_mask[10, 10] = 1
    _write_label_tiff(label_dir / filename, {"Nuclei": nuclei_array, "Spots": spots_mask})

    _write_metadata_csv(
        tiff_dir,
        nd2_stem,
        rows=[{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}],
    )

    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    convert_tiff_well_to_ome_ngff(
        nd2_stem=nd2_stem,
        tiff_dir=tiff_dir,
        plate_path=plate_path,
        label_dir=label_dir,
        point_object_channel_names=["Spots"],
    )

    container = open_ome_zarr_container(str(plate_path / "C" / "09" / "mip"))
    assert container.list_labels() == ["Nuclei"]
    assert "Spots" in container.list_tables()
    assert len(container.get_table("Spots").rois()) == 2


def test_convert_tiff_well_to_ome_ngff_raises_for_unknown_point_object_channel_name(tmp_path):
    nd2_stem = "WellC09_Seq0001"
    tiff_dir = tmp_path / "intensity"
    tiff_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()

    filename = f"{nd2_stem}_0001.ome.tiff"
    _write_blank_tiff(tiff_dir / filename)
    _write_label_tiff(label_dir / filename, {"Nuclei": np.zeros((16, 16), dtype="uint16")})
    _write_metadata_csv(
        tiff_dir,
        nd2_stem,
        rows=[{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}],
    )

    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    with pytest.raises(ValueError, match="Spots"):
        convert_tiff_well_to_ome_ngff(
            nd2_stem=nd2_stem,
            tiff_dir=tiff_dir,
            plate_path=plate_path,
            label_dir=label_dir,
            point_object_channel_names=["Spots"],
        )


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
        label_dir=fixture_dir / "SEGMENTATION",
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
