"""Tests for convert_nd2.py -- PBS jobscript generation for both output
formats ("TIFF" via nd2_to_ome_tiff, "NGFF" via nd2_to_ome_ngff)."""

from ngio import open_ome_zarr_plate
import pytest

from blimp.preprocessing.convert_nd2 import (
    convert_nd2,
    generate_pbs_script,
    generate_pbs_script_ngff,
)


def test_generate_pbs_script_formats_tiff_template():
    template = (
        "{INPUT_DIR}|{OUTPUT_DIR}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}|"
        "{MIP}|{KEEP_STACKS}|{Y_DIRECTION}|{CHANNEL_NAMES}"
    )
    result = generate_pbs_script(
        template=template,
        input_dir="/in",
        output_dir="/out",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=4,
        mip=True,
        keep_stacks=False,
        y_direction="down",
        channel_names=["DAPI", "GFP"],
    )
    assert result == (
        "/in|/out|/log|z1234567|a@b.com|4|#PBS -J 0-3|${PBS_ARRAY_INDEX}|--mip||down|--channel_names DAPI GFP"
    )


def test_generate_pbs_script_formats_tiff_template_skips_array_directive_for_one_batch():
    """Regression: a real PBS Pro cluster rejected "-J 0-0" outright as an
    illegal value, so a single-batch job must not use array syntax at all."""
    template = "{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}"
    result = generate_pbs_script(
        template=template,
        input_dir="/in",
        output_dir="/out",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=1,
        mip=True,
        keep_stacks=False,
        y_direction="down",
    )
    assert result == "|0"


def test_generate_pbs_script_ngff_formats_ngff_template():
    template = (
        "{INPUT_DIR}|{PLATE_PATH}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}|"
        "{MIP}|{KEEP_STACKS}|{Y_DIRECTION}|{X_DIRECTION}|{PLACEMENT}|{CHANNEL_NAMES}"
    )
    result = generate_pbs_script_ngff(
        template=template,
        input_dir="/in",
        plate_path="/plate.zarr",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=4,
        mip=True,
        keep_stacks=False,
        y_direction="down",
        x_direction="left",
        placement="grid",
        channel_names=None,
    )
    assert result == ("/in|/plate.zarr|/log|z1234567|a@b.com|4|#PBS -J 0-3|${PBS_ARRAY_INDEX}|--mip||down|left|grid|")


@pytest.fixture
def nd2_source_dir(tmp_path):
    source_dir = tmp_path / "WellA01"
    source_dir.mkdir()
    (source_dir / "WellA01.nd2").write_bytes(b"")
    return source_dir


def test_convert_nd2_rejects_unknown_image_format(nd2_source_dir, tmp_path):
    with pytest.raises(NotImplementedError, match='"TIFF" or "NGFF"'):
        convert_nd2(in_path=nd2_source_dir, job_path=tmp_path / "jobs", image_format="OME-ZARR")


def test_convert_nd2_tiff_writes_jobscript_and_ome_tiff_output_path(nd2_source_dir, tmp_path):
    job_path = tmp_path / "jobs"
    convert_nd2(in_path=nd2_source_dir, job_path=job_path, image_format="TIFF", n_batches=2)

    jobscript_path = job_path / f"batch_convert_nd2_{nd2_source_dir.stem}.pbs"
    assert jobscript_path.exists()
    content = jobscript_path.read_text()
    assert f'OUTPUT_DIR="{(nd2_source_dir / "OME-TIFF").resolve()}"' in content
    assert "nd2_to_ome_tiff.py" in content
    assert "--batch 2" in content


def test_convert_nd2_ngff_writes_jobscript_and_creates_plate(nd2_source_dir, tmp_path):
    job_path = tmp_path / "jobs"
    convert_nd2(
        in_path=nd2_source_dir,
        job_path=job_path,
        image_format="NGFF",
        n_batches=2,
        x_direction="right",
        placement="exact",
    )

    jobscript_path = job_path / f"batch_convert_nd2_{nd2_source_dir.stem}.pbs"
    assert jobscript_path.exists()
    content = jobscript_path.read_text()
    plate_path = nd2_source_dir / "plate.zarr"
    assert f'PLATE_PATH="{plate_path.resolve()}"' in content
    assert "nd2_to_ome_ngff.py convert" in content
    assert "-x right" in content
    assert "--placement exact" in content

    # ensure_plate_exists is called up front (job-generation time), not
    # deferred to the PBS array job itself -- so the plate skeleton must
    # already exist on disk without ever running the generated jobscript.
    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    assert plate.rows == [chr(c) for c in range(ord("A"), ord("A") + 16)]
