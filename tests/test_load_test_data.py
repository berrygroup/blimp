"""Tests for blimp.data.load_test_data -- recognising an out-of-date local
copy of the reference dataset, without touching the network."""
import os
import shutil

from blimp.data import load_test_data, is_test_data_complete
import blimp.data


def _make_dataset_zip(tmp_path, name, paths):
    """Zip a ``_data``/``_experiments`` tree containing ``paths``."""
    src = tmp_path / name
    (src / "_experiments").mkdir(parents=True)
    for p in paths:
        (src / "_data" / p).parent.mkdir(parents=True, exist_ok=True)
        (src / "_data" / p).touch()
    return shutil.make_archive(str(src), "zip", root_dir=src)


def _setup(tmp_path, monkeypatch, local_paths):
    """Point blimp.data at ``tmp_path`` with an extracted local ``_data.zip``
    containing ``local_paths``, and a fake Figshare serving the current
    dataset. Returns the list of download calls made."""
    tests_dir = tmp_path / "repo" / "tests"
    tests_dir.mkdir(parents=True)
    old_zip = _make_dataset_zip(tmp_path, "local", local_paths)
    shutil.copy(old_zip, tests_dir / "_data.zip")
    shutil.unpack_archive(tests_dir / "_data.zip", tests_dir)

    current_zip = _make_dataset_zip(tmp_path, "figshare", blimp.data.TEST_DATA_REQUIRED_PATHS)
    downloads = []

    def fake_download(repository_id, output_filename):
        downloads.append(repository_id)
        shutil.copy(current_zip, output_filename)
        return output_filename

    monkeypatch.setattr(blimp.data, "SCRIPTS_DIR", str(tmp_path / "repo"))
    monkeypatch.setattr(blimp.data, "download_figshare_file", fake_download)
    return tests_dir, downloads


def test_load_test_data_uses_current_local_copy(tmp_path, monkeypatch):
    tests_dir, downloads = _setup(tmp_path, monkeypatch, blimp.data.TEST_DATA_REQUIRED_PATHS)
    assert load_test_data() == str(tests_dir / "_data")
    assert downloads == []


def test_load_test_data_replaces_out_of_date_copy(tmp_path, monkeypatch):
    tests_dir, downloads = _setup(tmp_path, monkeypatch, [os.path.join("resources", "affine.txt")])
    assert not is_test_data_complete(tests_dir / "_data")
    load_test_data()
    assert downloads == ["23972244"]
    assert is_test_data_complete(tests_dir / "_data")
