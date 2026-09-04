"""Tests for convert_tiff.py -- PBS jobscript generation for
tiff_to_ome_ngff.py, plus its pre-flight label_dir/feature_csv_dir
validation."""
from pathlib import Path
import logging

from ngio import open_ome_zarr_plate
import pandas as pd
import pytest

from blimp.preprocessing.convert_tiff import convert_tiff, generate_pbs_script_tiff_ngff


def test_generate_pbs_script_tiff_ngff_formats_template_with_everything():
    template = (
        "{INPUT_DIR}|{PLATE_PATH}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{BATCH_MAX}|"
        "{Y_DIRECTION}|{X_DIRECTION}|{PLACEMENT}|{CHANNEL_NAMES}|{LABEL_DIR}|{FEATURE_CSV_DIR}|"
        "{POINT_OBJECT_CHANNEL_NAMES}"
    )
    result = generate_pbs_script_tiff_ngff(
        template=template,
        input_dir="/in",
        plate_path="/plate.zarr",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=4,
        y_direction="down",
        x_direction="left",
        placement="grid",
        channel_names=["DAPI", "GFP"],
        label_dir="/labels",
        feature_csv_dir="/features",
        point_object_channel_names=["Spots"],
    )
    assert result == (
        "/in|/plate.zarr|/log|z1234567|a@b.com|4|3|down|left|grid|--channel_names DAPI GFP|"
        "--label_dir /labels|--feature_csv_dir /features|--point_object_channel_names Spots"
    )


def test_generate_pbs_script_tiff_ngff_formats_template_with_nothing_optional():
    template = "{LABEL_DIR}|{FEATURE_CSV_DIR}|{CHANNEL_NAMES}|{POINT_OBJECT_CHANNEL_NAMES}"
    result = generate_pbs_script_tiff_ngff(
        template=template,
        input_dir="/in",
        plate_path="/plate.zarr",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=1,
        y_direction="down",
        x_direction="left",
        placement="grid",
    )
    assert result == "|||"


@pytest.fixture
def tiff_pipeline_dir(tmp_path):
    in_path = tmp_path / "OME-TIFF-MIP"
    in_path.mkdir()
    filename = "WellA01_0001.ome.tiff"
    pd.DataFrame([{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}]).to_csv(
        in_path / "WellA01_metadata.csv", index=False
    )
    return in_path, filename


def test_convert_tiff_rejects_unknown_image_format(tiff_pipeline_dir, tmp_path):
    in_path, _ = tiff_pipeline_dir
    with pytest.raises(NotImplementedError, match="NGFF"):
        convert_tiff(in_path=in_path, plate_path=tmp_path / "plate.zarr", image_format="TIFF")


def test_convert_tiff_raises_when_no_wells_found(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        convert_tiff(in_path=empty_dir, plate_path=tmp_path / "plate.zarr")


def test_convert_tiff_warns_when_label_dir_and_feature_csv_dir_omitted(tiff_pipeline_dir, tmp_path, caplog):
    in_path, _ = tiff_pipeline_dir
    with caplog.at_level(logging.WARNING):
        convert_tiff(in_path=in_path, plate_path=tmp_path / "plate.zarr", job_path=tmp_path / "jobs")
    assert "No label_dir given" in caplog.text
    assert "No feature_csv_dir given" in caplog.text


def test_convert_tiff_raises_for_label_dir_with_no_matching_files(tiff_pipeline_dir, tmp_path):
    in_path, _ = tiff_pipeline_dir
    empty_label_dir = tmp_path / "labels"
    empty_label_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="label_dir"):
        convert_tiff(
            in_path=in_path,
            plate_path=tmp_path / "plate.zarr",
            job_path=tmp_path / "jobs",
            label_dir=empty_label_dir,
        )


def test_convert_tiff_raises_for_feature_csv_dir_with_no_matching_files(tiff_pipeline_dir, tmp_path):
    in_path, _ = tiff_pipeline_dir
    empty_feature_dir = tmp_path / "features"
    empty_feature_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="feature_csv_dir"):
        convert_tiff(
            in_path=in_path,
            plate_path=tmp_path / "plate.zarr",
            job_path=tmp_path / "jobs",
            feature_csv_dir=empty_feature_dir,
        )


def test_convert_tiff_proceeds_with_partial_label_coverage(tiff_pipeline_dir, tmp_path):
    """A label_dir with *some* matching files is fine -- per-field gaps are
    a warning (already logged by _discover_well_manifest itself), not an
    error."""
    in_path, filename = tiff_pipeline_dir
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / filename).write_bytes(b"")  # just needs to exist for this check

    job_path = tmp_path / "jobs"
    convert_tiff(in_path=in_path, plate_path=tmp_path / "plate.zarr", job_path=job_path, label_dir=label_dir)
    assert (job_path / f"batch_convert_tiff_{in_path.stem}.pbs").exists()


def test_convert_tiff_writes_jobscript_and_creates_plate(tiff_pipeline_dir, tmp_path):
    in_path, filename = tiff_pipeline_dir
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / filename).write_bytes(b"")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / f"{Path(filename).stem}.csv").write_bytes(b"")

    job_path = tmp_path / "jobs"
    plate_path = tmp_path / "plate.zarr"
    convert_tiff(
        in_path=in_path,
        plate_path=plate_path,
        job_path=job_path,
        label_dir=label_dir,
        feature_csv_dir=feature_dir,
        n_batches=2,
        x_direction="right",
        placement="exact",
    )

    jobscript_path = job_path / f"batch_convert_tiff_{in_path.stem}.pbs"
    assert jobscript_path.exists()
    content = jobscript_path.read_text()
    assert f'PLATE_PATH="{plate_path.resolve()}"' in content
    assert "tiff_to_ome_ngff.py convert" in content
    assert "-x right" in content
    assert "--placement exact" in content
    assert f"--label_dir {label_dir.resolve()}" in content
    assert f"--feature_csv_dir {feature_dir.resolve()}" in content

    # ensure_plate_exists is called up front (job-generation time), not
    # deferred to the PBS array job itself -- so the plate skeleton must
    # already exist on disk without ever running the generated jobscript.
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    assert plate.rows == [chr(c) for c in range(ord("A"), ord("A") + 16)]


def test_convert_tiff_twice_with_same_plate_path_different_in_paths(tmp_path):
    """The MIP+STACKS pattern from review: calling convert_tiff twice with
    the same explicit plate_path but two different in_paths (e.g. an
    OME-TIFF-MIP/ source and an OME-TIFF/ source) must both succeed."""
    plate_path = tmp_path / "plate.zarr"
    job_path = tmp_path / "jobs"

    for name in ["OME-TIFF-MIP", "OME-TIFF"]:
        in_path = tmp_path / name
        in_path.mkdir()
        filename = "WellA01_0001.ome.tiff"
        pd.DataFrame([{"field_id": 1, "stage_x_abs": 0.0, "stage_y_abs": 0.0, "filename_ome_tiff": filename}]).to_csv(
            in_path / "WellA01_metadata.csv", index=False
        )
        convert_tiff(in_path=in_path, plate_path=plate_path, job_path=job_path)
        assert (job_path / f"batch_convert_tiff_{name}.pbs").exists()
