"""Tests for blimp.ome_ngff.plate -- shared OME-Zarr plate/well registration
and image writing."""
import zarr
import pytest

from blimp.ome_ngff.plate import open_well_image, ensure_plate_exists


def test_open_well_image_raises_clear_error_for_missing_kind(tmp_path):
    """No multi-position .nd2 fixture exists to exercise the full write
    pipeline end-to-end, so this builds just enough of a well's on-disk
    structure by hand -- a "mip" image directory plus the well-level
    ``ome.well.images`` metadata a real conversion would produce -- to
    exercise open_well_image's validation in isolation."""
    plate_path = tmp_path / "plate.zarr"
    well_relative_path = "C/09"
    well_path = plate_path / well_relative_path
    (well_path / "mip").mkdir(parents=True)

    well_group = zarr.open_group(str(well_path), mode="a", zarr_format=3)
    well_group.attrs["ome"] = {
        "version": "0.5",
        "well": {"version": "0.5", "images": [{"path": "mip"}]},
    }

    with pytest.raises(FileNotFoundError) as excinfo:
        open_well_image(plate_path, well_relative_path, kind="stack")
    message = str(excinfo.value)
    assert "'stack'" in message
    assert "'mip'" in message
    assert "keep_stacks=True" in message


def test_ensure_plate_exists_predeclares_full_384_well_grid(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    plate = ensure_plate_exists(plate_path, "test_plate")
    assert plate.rows == [chr(ord("A") + i) for i in range(16)]
    assert plate.columns == [f"{i:02d}" for i in range(1, 25)]

    # pre-declaring the grid costs nothing in storage: nothing exists yet
    # besides the plate's own zarr.json until a well is actually written.
    assert [p.name for p in plate_path.iterdir()] == ["zarr.json"]


def test_ensure_plate_exists_predeclares_96_well_grid_when_requested(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    plate = ensure_plate_exists(plate_path, "test_plate", plate_size="96")
    assert plate.rows == [chr(ord("A") + i) for i in range(8)]
    assert plate.columns == [f"{i:02d}" for i in range(1, 13)]


def test_ensure_plate_exists_is_idempotent(tmp_path):
    plate_path = tmp_path / "plate.zarr"
    ensure_plate_exists(plate_path, "test_plate")
    plate2 = ensure_plate_exists(plate_path, "test_plate")
    assert plate2.rows == [chr(ord("A") + i) for i in range(16)]
