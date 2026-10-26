"""Tests for convert_opera.py -- PBS jobscript generation and conversion
checks for Opera Phenix exports."""

import pytest

from blimp.preprocessing.convert_opera import (
    convert_opera,
    find_images_dirs,
    generate_pbs_script,
    check_convert_opera_dir,
)
from blimp.preprocessing.opera_to_ome_tiff import opera_to_ome_tiff

from .helpers import _make_opera_dataset


def test_generate_pbs_script_formats_template():
    template = (
        "{INPUT_DIR}|{LOG_DIR}|{USER}|{USER_EMAIL}|{N_BATCHES}|{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}|"
        "{MIP}|{KEEP_STACKS}|{SAVE_METADATA_FILES}|{CONDA_ENV}"
    )
    result = generate_pbs_script(
        template=template,
        input_dir="/in",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=4,
        mip=True,
        keep_stacks=False,
        save_metadata_files=True,
        conda_env="env",
    )
    assert result == ("/in|/log|z1234567|a@b.com|4|#PBS -J 0-3|${PBS_ARRAY_INDEX}|--mip||--save_metadata_files|env")


def test_generate_pbs_script_skips_array_directive_for_one_batch():
    """PBS Pro rejects "-J 0-0" as an illegal array-job value, so a
    single-batch job must not use array syntax at all."""
    result = generate_pbs_script(
        template="{ARRAY_DIRECTIVE}|{BATCH_ID_EXPR}",
        input_dir="/in",
        log_dir="/log",
        user="z1234567",
        email="a@b.com",
        n_batches=1,
        mip=True,
        keep_stacks=False,
        save_metadata_files=True,
    )
    assert result == "|0"


@pytest.fixture
def measurement_dir(tmp_path):
    measurement_dir = tmp_path / "experiment" / "plate1" / "7ff28d40-ab05-4813-b326-50cfde35c09d"
    _make_opera_dataset(measurement_dir)
    return measurement_dir


def test_find_images_dirs_requires_index_xml(measurement_dir, tmp_path):
    (tmp_path / "experiment" / "other" / "images").mkdir(parents=True)
    assert find_images_dirs(tmp_path) == [str(measurement_dir / "images")]


def test_convert_opera_rejects_unknown_image_format(measurement_dir, tmp_path):
    with pytest.raises(NotImplementedError, match='"TIFF"'):
        convert_opera(in_path=measurement_dir, job_path=tmp_path / "jobs", image_format="NGFF")


def test_convert_opera_writes_jobscript(measurement_dir, tmp_path):
    job_path = tmp_path / "jobs"
    convert_opera(in_path=tmp_path, job_path=job_path, image_format="TIFF", n_batches=2, mip=True)

    jobscript_path = job_path / f"batch_convert_opera_{measurement_dir.name}.pbs"
    assert jobscript_path.exists()
    assert (job_path / "log").is_dir()
    content = jobscript_path.read_text()
    assert f'INPUT_DIR="{measurement_dir.resolve()}"' in content
    assert f'OUTPUT_DIR="{measurement_dir.resolve()}/OME-TIFF"' in content
    assert "opera_to_ome_tiff.py" in content
    assert '-i "$INPUT_DIR/images"' in content
    assert "--batch 2" in content
    assert "--mip" in content


def test_check_convert_opera_dir_raises_without_output(measurement_dir):
    with pytest.raises(FileNotFoundError):
        check_convert_opera_dir(measurement_dir / "images")


def test_check_convert_opera_dir_reports_missing_then_complete(measurement_dir):
    images_dir = measurement_dir / "images"

    # convert only the first of two batches
    opera_to_ome_tiff(in_path=images_dir, out_path=measurement_dir / "OME-TIFF", n_batches=2, batch_id=0, mip=True)
    missing = check_convert_opera_dir(images_dir, mip=True, keep_stacks=False)
    assert missing == {(3, 6, 1), (3, 6, 2)}

    opera_to_ome_tiff(in_path=images_dir, out_path=measurement_dir / "OME-TIFF", n_batches=2, batch_id=1, mip=True)
    assert check_convert_opera_dir(images_dir, mip=True, keep_stacks=False) == set()

    # stacks were never written
    assert len(check_convert_opera_dir(images_dir, mip=True, keep_stacks=True)) == 4
