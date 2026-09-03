"""Tests for blimp.ome_ngff.features -- feature-table attachment with the
same per-field global-ID offset used for label placement."""
from pathlib import Path

import ngio
import numpy as np
import pandas as pd

from blimp.ome_ngff.labels import _write_well_labels, MAX_OBJECTS_PER_FIELD
from blimp.ome_ngff.layout import FieldLayout
from blimp.ome_ngff.features import _write_well_features, _offset_feature_table_ids


def _make_field_layout(**overrides):
    defaults = dict(
        row="C",
        column=9,
        offsets=[(0, 0), (0, 16)],
        tile_shape=(1, 1, 1, 16, 16),
        canvas_shape=(1, 1, 1, 16, 32),
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        pixel_size_z=1.0,
        channel_names=["DAPI"],
        channel_colors=["0000FF"],
        position_names=["C09_0001", "C09_0002"],
        field_ids=[1, 2],
    )
    defaults.update(overrides)
    return FieldLayout(**defaults)


def _make_container(tmp_path: Path):
    store = str(tmp_path / "plate.zarr")
    return ngio.create_ome_zarr_from_array(
        store=store,
        array=np.zeros((1, 1, 1, 16, 32), dtype=np.uint16),
        pixelsize=0.5,
        axes_names=["t", "c", "z", "y", "x"],
        levels=2,
        ngff_version="0.5",
    )


def test_offset_feature_table_ids_shifts_label_and_parent_label():
    df = pd.DataFrame({"label": [1, 2], "parent_label": [1, 1], "area": [10, 20]})
    out = _offset_feature_table_ids(df, field_id=3, well_name="C09")
    offset = 3 * MAX_OBJECTS_PER_FIELD
    assert out["label"].tolist() == [1 + offset, 2 + offset]
    assert out["parent_label"].tolist() == [1 + offset, 1 + offset]
    assert out["area"].tolist() == [10, 20]


def test_offset_feature_table_ids_adds_human_readable_fov_object_id():
    """The numeric label/parent_label offset has no way to carry the well
    name (it's an integer pixel value) -- fov_object_id is the traceability
    companion, built from the same (well, field_id, local_id), so it should
    never disagree with the numeric global_id."""
    df = pd.DataFrame({"label": [1, 2]})
    out = _offset_feature_table_ids(df, field_id=4, well_name="C09")
    assert out["fov_object_id"].tolist() == ["C09_0004_0000001", "C09_0004_0000002"]


def test_offset_feature_table_ids_does_not_mutate_input():
    df = pd.DataFrame({"label": [1]})
    _offset_feature_table_ids(df, field_id=1, well_name="C09")
    assert df["label"].tolist() == [1]


def test_write_well_features_matches_label_ids(tmp_path):
    container = _make_container(tmp_path)
    layout = _make_field_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    field0[10:14, 10:14] = 2
    field1 = np.zeros((16, 16), dtype=np.uint16)
    field1[2:6, 2:6] = 1

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: field1})

    features0 = pd.DataFrame({"label": [1, 2], "Nuclei_area": [16, 16]})
    features1 = pd.DataFrame({"label": [1], "Nuclei_area": [16]})
    _write_well_features(
        container=container,
        label_name="Nuclei",
        field_dataframes={1: features0, 2: features1},
        well_name="C09",
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    label_ids = set(np.unique(container.get_label("Nuclei").get_as_numpy()).tolist()) - {0}
    assert set(df["label"].tolist()) == label_ids
    assert set(df["fov_object_id"].tolist()) == {"C09_0001_0000001", "C09_0001_0000002", "C09_0002_0000001"}


def test_write_well_features_field_with_missing_measurements_contributes_no_rows(tmp_path):
    container = _make_container(tmp_path)
    layout = _make_field_layout()

    field0 = np.zeros((16, 16), dtype=np.uint16)
    field0[2:6, 2:6] = 1
    field1 = np.zeros((16, 16), dtype=np.uint16)
    field1[2:6, 2:6] = 1

    _write_well_labels(container=container, layout=layout, label_name="Nuclei", field_arrays={1: field0, 2: field1})

    features0 = pd.DataFrame({"label": [1], "Nuclei_area": [16]})
    # field 2's measurements are missing entirely (e.g. quantification failed)
    _write_well_features(
        container=container, label_name="Nuclei", field_dataframes={1: features0, 2: None}, well_name="C09"
    )

    table = container.get_feature_table("Nuclei_features")
    df = table.dataframe.reset_index()
    assert len(df) == 1
    assert df["label"].iloc[0] == 1 * MAX_OBJECTS_PER_FIELD + 1
