"""Tests for blimp.data.load_test_data -- recognising an out-of-date local
copy of the reference dataset. Downloading is not tested here: CI's testdata
job downloads the dataset from Figshare on every run."""
import os
import shutil

from blimp.data import load_test_data, is_test_data_complete, TEST_DATA_REQUIRED_PATHS


def _write_dataset(base_dir, paths):
    """Write a ``_data``/``_experiments`` tree with an empty file at each of
    ``paths`` (relative to ``_data``)."""
    (base_dir / "_experiments").mkdir(parents=True, exist_ok=True)
    for p in paths:
        (base_dir / "_data" / p).parent.mkdir(parents=True, exist_ok=True)
        (base_dir / "_data" / p).touch()


def test_is_test_data_complete(tmp_path):
    _write_dataset(tmp_path, [os.path.join("resources", "affine.txt")])
    assert not is_test_data_complete(tmp_path / "_data")
    _write_dataset(tmp_path, TEST_DATA_REQUIRED_PATHS)
    assert is_test_data_complete(tmp_path / "_data")


def test_load_test_data_uses_current_extracted_copy(tmp_path):
    _write_dataset(tmp_path, TEST_DATA_REQUIRED_PATHS)
    assert load_test_data(base_dir=tmp_path) == str(tmp_path / "_data")


def test_load_test_data_re_extracts_out_of_date_copy(tmp_path):
    # a current _data.zip next to an older extraction, as left by a manual
    # download of a new dataset version
    _write_dataset(tmp_path / "current", TEST_DATA_REQUIRED_PATHS)
    shutil.make_archive(str(tmp_path / "_data"), "zip", root_dir=tmp_path / "current")
    _write_dataset(tmp_path, [os.path.join("resources", "affine.txt")])

    load_test_data(base_dir=tmp_path)
    assert is_test_data_complete(tmp_path / "_data")
