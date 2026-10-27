"""Tests for convert_operetta.py -- PBS jobscript generation and conversion
checks for Operetta exports."""

from ngio import open_ome_zarr_plate
import pytest

from blimp.preprocessing.convert_operetta import (
    convert_operetta,
    generate_pbs_script,
    check_convert_operetta,
    check_convert_operetta_dir,
)
from blimp.preprocessing.harmony_to_ome_ngff import harmony_to_ome_ngff
from blimp.preprocessing.operetta_to_ome_tiff import operetta_to_ome_tiff

from .helpers import _make_operetta_dataset


def test_generate_pbs_script_formats_template():
    template = (
        "{INPUT_DIR}|{OUTPUT_DIR}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}|"
        "{MIP}|{KEEP_STACKS}|{SAVE_METADATA_FILES}|{CONDA_ENV}"
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
        save_metadata_files=True,
        conda_env="env",
    )
    assert result == (
        "/in|/out|/log|z1234567|a@b.com|4|#PBS -J 0-3|${PBS_ARRAY_INDEX}|--mip||--save_metadata_files|env"
    )


@pytest.fixture
def experiment_dir(tmp_path):
    experiment_dir = tmp_path / "experiment"
    _make_operetta_dataset(experiment_dir)
    return experiment_dir


def test_convert_operetta_writes_jobscript(experiment_dir, tmp_path):
    job_path = tmp_path / "jobs"
    convert_operetta(in_path=tmp_path, job_path=job_path, image_format="TIFF", n_batches=2, mip=True)

    content = (job_path / "batch_convert_operetta_experiment.pbs").read_text()
    assert f'INPUT_DIR="{experiment_dir.resolve()}"' in content
    assert "operetta_to_ome_tiff.py" in content
    assert "--batch 2" in content


def test_check_convert_operetta_dir_reports_missing_then_complete(experiment_dir):
    images_dir = experiment_dir / "Images"
    out_path = experiment_dir / "OME-TIFF"

    convert = dict(in_path=images_dir, out_path=out_path, metadata_file="Index.idx.xml", n_batches=2, keep_stacks=True)
    operetta_to_ome_tiff(batch_id=0, **convert)
    assert check_convert_operetta_dir(images_dir) == {("03", "06", "01"), ("03", "06", "02")}

    operetta_to_ome_tiff(batch_id=1, **convert)
    assert check_convert_operetta_dir(images_dir) == set()


def test_check_convert_operetta_dir_ignores_digits_in_parent_dirs(tmp_path):
    # e.g. macOS temporary directories such as /var/folders/.../f2113800000/
    experiment_dir = tmp_path / "rc1" / "experiment"
    _make_operetta_dataset(experiment_dir)
    images_dir = experiment_dir / "Images"
    operetta_to_ome_tiff(
        in_path=images_dir,
        out_path=experiment_dir / "OME-TIFF",
        metadata_file="Index.idx.xml",
        n_batches=2,
        batch_id=0,
        keep_stacks=True,
    )
    assert check_convert_operetta_dir(images_dir) == {("03", "06", "01"), ("03", "06", "02")}


def test_convert_operetta_writes_to_users_own_directory(tmp_path):
    """Source data under another user's zID is converted into the same
    relative path under the converting user's zID, without writing to the
    source directory."""
    source_dir = tmp_path / "z1111111" / "operetta_cls" / "experiment"
    _make_operetta_dataset(source_dir)
    for p in [source_dir, *source_dir.rglob("*")]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    out_path = tmp_path / "z2222222" / "operetta_cls" / "experiment" / "OME-TIFF"
    try:
        convert_operetta(
            in_path=tmp_path / "z1111111", job_path=tmp_path / "jobs", image_format="TIFF", user="z2222222"
        )
        content = (tmp_path / "jobs" / "batch_convert_operetta_experiment.pbs").read_text()
        assert f'OUTPUT_DIR="{out_path.resolve()}"' in content

        operetta_to_ome_tiff(
            in_path=source_dir / "Images", out_path=out_path, metadata_file="Index.idx.xml", keep_stacks=True
        )
        assert (out_path.parent / "image_metadata.csv").exists()
        assert check_convert_operetta_dir(source_dir / "Images", user="z2222222") == set()
    finally:
        for p in [source_dir, *source_dir.rglob("*")]:
            p.chmod(0o755)


def test_convert_operetta_rejects_unknown_image_format(experiment_dir, tmp_path):
    with pytest.raises(NotImplementedError, match='"TIFF" or "NGFF"'):
        convert_operetta(in_path=tmp_path, job_path=tmp_path / "jobs", image_format="OME-ZARR")


def test_convert_operetta_ngff_writes_to_users_own_directory(tmp_path, caplog):
    source_dir = tmp_path / "z1111111" / "operetta_cls" / "experiment"
    _make_operetta_dataset(source_dir)
    for p in [source_dir, *source_dir.rglob("*")]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    plate_path = tmp_path / "z2222222" / "operetta_cls" / "experiment" / "plate.zarr"
    try:
        convert_operetta(
            in_path=tmp_path / "z1111111",
            job_path=tmp_path / "jobs",
            image_format="NGFF",
            mip=True,
            placement="exact",
            channel_names=["DAPI", "GFP"],
            user="z2222222",
        )
        content = (tmp_path / "jobs" / "batch_convert_operetta_experiment.pbs").read_text()
        assert f'PLATE_PATH="{plate_path.resolve()}"' in content
        assert '-i "$INPUT_DIR/Images"' in content
        assert "--instrument operetta" in content
        assert "--placement exact" in content
        assert "--channel_names DAPI GFP" in content
        assert open_ome_zarr_plate(store=str(plate_path), mode="r").rows == list("ABCDEFGH")

        harmony_to_ome_ngff(source_dir / "Images", plate_path, "operetta", mip=True)
        assert (plate_path.parent / "image_metadata.csv").exists()
        with caplog.at_level("INFO"):
            check_convert_operetta(
                tmp_path / "z1111111", mip=True, keep_stacks=False, user="z2222222", image_format="NGFF"
            )
        assert "Conversion appears complete" in caplog.text
    finally:
        for p in [source_dir, *source_dir.rglob("*")]:
            p.chmod(0o755)
