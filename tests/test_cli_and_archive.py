from pathlib import Path

import pytest

from blimp.archive import (
    archive,
    write_archiving_script_nd2,
    write_archiving_batch_files,
    write_archiving_script_operetta,
    split_opera_files_into_archiving_batches,
    split_operetta_files_into_archiving_batches,
)
from blimp.cli.main import _get_full_parser

from .helpers import _make_opera_dataset


def _parse(argv):
    return _get_full_parser().parse_args(argv)


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        _parse([])


def test_parser_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        _parse(["not-a-command"])


def test_convert_nd2_minimal_invocation(tmp_path):
    args = _parse(["convert", "nd2", "-i", str(tmp_path), "--user", "z1234567"])
    assert args.subcommand == "convert"
    assert args.input_type == "nd2"
    assert args.user == "z1234567"
    assert callable(args.func)


def test_convert_nd2_requires_user():
    with pytest.raises(SystemExit):
        _parse(["convert", "nd2", "-i", "."])


def test_convert_operetta_minimal_invocation(tmp_path):
    args = _parse(["convert", "operetta", "-i", str(tmp_path), "--user", "z1234567"])
    assert args.input_type == "operetta"
    assert callable(args.func)


def test_convert_opera_minimal_invocation(tmp_path):
    args = _parse(["convert", "opera", "-i", str(tmp_path), "--user", "z1234567"])
    assert args.input_type == "opera"
    assert args.output_format == "TIFF"
    assert callable(args.func)


def test_convert_opera_accepts_verify_only_with_mip(tmp_path):
    args = _parse(["convert", "opera", "-i", str(tmp_path), "--user", "z1234567", "--verify-only", "--mip"])
    assert args.verify_only is True
    assert args.mip is True


def test_convert_requires_input_type():
    with pytest.raises(SystemExit):
        _parse(["convert", "-i", "."])


def test_convert_tiff_minimal_invocation(tmp_path):
    args = _parse(["convert", "tiff", "-i", str(tmp_path), "-o", str(tmp_path / "plate.zarr"), "--user", "z1234567"])
    assert args.input_type == "tiff"
    assert callable(args.func)


def test_convert_tiff_requires_user(tmp_path):
    with pytest.raises(SystemExit):
        _parse(["convert", "tiff", "-i", str(tmp_path), "-o", str(tmp_path / "plate.zarr")])


def test_convert_tiff_requires_plate_path(tmp_path):
    with pytest.raises(SystemExit):
        _parse(["convert", "tiff", "-i", str(tmp_path), "--user", "z1234567"])


def test_convert_tiff_accepts_label_and_feature_flags(tmp_path):
    args = _parse(
        [
            "convert",
            "tiff",
            "-i",
            str(tmp_path),
            "-o",
            str(tmp_path / "plate.zarr"),
            "--user",
            "z1234567",
            "-l",
            "/labels",
            "-f",
            "/features",
            "--point_object_channel_names",
            "Spots",
            "Blobs",
        ]
    )
    assert args.label_dir == "/labels"
    assert args.feature_csv_dir == "/features"
    assert args.point_object_channel_names == ["Spots", "Blobs"]


def test_convert_tiff_output_format_defaults_to_ngff(tmp_path):
    args = _parse(["convert", "tiff", "-i", str(tmp_path), "-o", str(tmp_path / "plate.zarr"), "--user", "z1234567"])
    assert args.output_format == "NGFF"


@pytest.mark.parametrize("input_type", ["nd2", "operetta", "opera"])
def test_archive_subcommands_parse(input_type, tmp_path):
    """-j/--jobscript_path must parse with no explicit value given (default
    ``None``, resolved to cwd at call time), not be required."""
    args = _parse(["archive", input_type, "-i", str(tmp_path), "--first_name", "Ada"])
    assert args.subcommand == "archive"
    assert args.input_type == input_type
    assert args.jobscript_path is None  # resolved to cwd at call time
    assert callable(args.func)


def test_archive_accepts_explicit_jobscript_path(tmp_path):
    args = _parse(["archive", "nd2", "-i", str(tmp_path), "--first_name", "Ada", "-j", str(tmp_path)])
    assert args.jobscript_path == str(tmp_path)


def test_setup_accepts_quiet_flag():
    """`blimp setup --quiet` must parse via the top-level parser -- --quiet
    is registered there, not only on a separate subparser."""
    args = _parse(["setup", "--quiet"])
    assert args.quiet is True


def test_setup_quiet_defaults_false():
    assert _parse(["setup"]).quiet is False


def test_verbose_flag_is_available_on_base_parser(tmp_path):
    args = _parse(["-vv", "convert", "nd2", "-i", str(tmp_path), "--user", "z1234567"])
    assert args.verbose == 2


def test_setup_subcommand_parses():
    args = _parse(["setup"])
    assert args.subcommand == "setup"
    assert callable(args.func)


def test_setup_namespace_exposes_everything_main_reads():
    """``main()`` reads args.verbose and args.quiet on the setup path."""
    args = _parse(["setup"])
    assert hasattr(args, "verbose")
    assert hasattr(args, "quiet")


def test_split_operetta_files_groups_by_row_column(tmp_path):
    names = [
        "r01c01f01p01-ch1sk1fk1fl1.tiff",
        "r01c01f02p01-ch1sk1fk1fl1.tiff",
        "r02c03f01p01-ch1sk1fk1fl1.tiff",
        "Index.idx.xml",
    ]
    for n in names:
        (tmp_path / n).touch()
    groups = split_operetta_files_into_archiving_batches(tmp_path)
    assert set(groups) == {"r01c01", "r02c03"}
    assert len(groups["r01c01"]) == 2
    assert len(groups["r02c03"]) == 1


def test_split_operetta_files_ignores_non_matching_names(tmp_path):
    (tmp_path / "README.txt").touch()
    assert split_operetta_files_into_archiving_batches(tmp_path) == {}


def test_write_archiving_batch_files_writes_one_file_per_group(tmp_path):
    images = tmp_path / "Images"
    archive = tmp_path / "Archive"
    images.mkdir()
    archive.mkdir()
    groups = {"r01c01": ["a.tiff", "b.tiff"], "r02c02": ["c.tiff"]}
    write_archiving_batch_files(archive, images, groups)

    first = archive / "r01c01.txt"
    assert first.exists()
    lines = first.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(str(images) in line for line in lines)
    assert (archive / "r02c02.txt").read_text().strip().endswith("c.tiff")


def test_write_archiving_batch_files_raises_for_unwritable_dir(tmp_path):
    groups = {"r01c01": ["a.tiff"]}
    with pytest.raises(FileNotFoundError):
        write_archiving_batch_files(tmp_path / "missing", tmp_path, groups)


def test_write_archiving_script_nd2_produces_runnable_header(tmp_path):
    script = tmp_path / "archive_nd2.sh"
    write_archiving_script_nd2(
        script_path=script,
        file_paths=[Path("/srv/scratch/berrylab/z1234567/experiment/file.nd2")],
        first_name="Ada",
        project_name="PROJ",
    )
    text = script.read_text()
    assert text.startswith("#!/bin/bash")
    assert "module add unswdataarchive" in text
    assert "PROJ" in text and "Ada" in text
    # the scratch prefix must be stripped from the archive namespace
    assert "/srv/scratch/berrylab/z1234567/" not in text.split("-namespace")[1].split("'")[0]


def test_write_archiving_script_operetta_without_archive_dir(tmp_path):
    """A file list with no 'Archive' directory must still produce a valid
    script, not raise ``UnboundLocalError`` for unset archive variables."""
    script = tmp_path / "archive_operetta.sh"
    write_archiving_script_operetta(
        script_path=script,
        file_paths=[Path("/srv/scratch/berrylab/z1234567/experiment/Images")],
        first_name="Ada",
        project_name="PROJ",
    )
    text = script.read_text()
    assert text.startswith("#!/bin/bash")
    assert "## Upload:" in text
    # no checksum section should be emitted when there is no Archive directory
    assert "Compute checksums locally" not in text


def test_write_archiving_script_operetta_with_archive_dir(tmp_path):
    archive_dir = tmp_path / "Archive"
    archive_dir.mkdir()
    (archive_dir / "r01c01.txt").write_text("dummy\n")
    script = tmp_path / "archive_operetta2.sh"
    write_archiving_script_operetta(
        script_path=script,
        file_paths=[archive_dir],
        first_name="Ada",
        project_name="PROJ",
    )
    text = script.read_text()
    assert "Compute checksums locally" in text
    assert "tar -cvzf" in text


def test_write_archiving_script_append_mode(tmp_path):
    script = tmp_path / "append.sh"
    paths = [Path("/srv/scratch/berrylab/z1234567/exp/Images")]
    write_archiving_script_operetta(paths, script, "Ada", "PROJ")
    first_len = len(script.read_text())
    write_archiving_script_operetta(paths, script, "Ada", "PROJ", append=True)
    text = script.read_text()
    assert len(text) > first_len
    # the shebang must appear only once
    assert text.count("#!/bin/bash") == 1


@pytest.fixture
def opera_measurement_dir(tmp_path):
    """Synthetic opera export, plus the metadata files of a real export and
    blimp-written outputs that should not be archived."""
    measurement_dir = tmp_path / "experiment" / "7ff28d40-ab05-4813-b326-50cfde35c09d"
    _make_opera_dataset(measurement_dir, wells=((3, 5), (3, 6)), field_positions=((0.0, 0.0),), n_planes=1)
    (measurement_dir / "7ff28d40-ab05-4813-b326.xml").touch()
    (measurement_dir / "flatfieldcorrection").mkdir()
    (measurement_dir / "flatfieldcorrection" / "155138c0-0062-4414-b93f.xml").touch()
    (measurement_dir / "images" / "image.index.txt").touch()
    for output in ["OME-TIFF/r03c05f01.ome.tiff", "OME-NGFF/plate.zarr/zarr.json", "plate.zarr/zarr.json"]:
        (measurement_dir / output).parent.mkdir(parents=True, exist_ok=True)
        (measurement_dir / output).touch()
    return measurement_dir


def test_split_opera_files_groups_by_well_plus_metadata(opera_measurement_dir):
    groups = split_opera_files_into_archiving_batches(opera_measurement_dir)
    assert set(groups) == {"r03c05", "r03c06", "metadata"}
    assert groups["r03c05"] == ["images/r03c05/r03c05f01p01-ch01t01.tiff", "images/r03c05/r03c05f01p01-ch02t01.tiff"]
    assert sorted(groups["metadata"]) == [
        "7ff28d40-ab05-4813-b326.xml",
        "flatfieldcorrection/155138c0-0062-4414-b93f.xml",
        "images/image.index.txt",
        "images/index.xml",
    ]


def test_split_opera_files_skips_existing_archive(opera_measurement_dir):
    (opera_measurement_dir / "Archive").mkdir()
    (opera_measurement_dir / "Archive" / "r03c05.txt").touch()
    groups = split_opera_files_into_archiving_batches(opera_measurement_dir)
    assert not any(f.startswith("Archive") for files in groups.values() for f in files)


def test_archive_opera_writes_batches_and_script(opera_measurement_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("blimp.archive.check_config_file", lambda: True)
    job_path = tmp_path / "jobs"
    job_path.mkdir()
    archive(
        in_path=tmp_path / "experiment",
        jobscript_path=job_path,
        first_name="Ada",
        project_name="PROJ",
        input_type="opera",
    )

    archive_dir = opera_measurement_dir / "Archive"
    assert sorted(p.name for p in archive_dir.glob("*.txt")) == ["metadata.txt", "r03c05.txt", "r03c06.txt"]
    batch = (archive_dir / "r03c06.txt").read_text().splitlines()
    assert batch == [str(opera_measurement_dir / "images" / "r03c06" / f"r03c06f01p01-ch0{c}t01.tiff") for c in (1, 2)]

    (script_path,) = job_path.glob("archive_data_blimp_*.sh")
    text = script_path.read_text()
    assert f"tar -cvzf '{archive_dir / 'r03c05.tar.gz'}' -T '{archive_dir / 'r03c05.txt'}'" in text
    assert f"'{archive_dir}'" in text
    assert "Compute checksums locally" in text
    assert "OME-TIFF" not in text
    assert "zarr" not in "".join(p.read_text() for p in archive_dir.glob("*.txt"))
