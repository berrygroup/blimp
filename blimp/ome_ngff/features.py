"""Feature-table attachment for OME-NGFF assembly: applies the same
per-field global-ID offset used for label placement (see
:mod:`blimp.ome_ngff.labels`) to a ``quantify()`` output's ``label``/
``parent_label`` columns, then concatenates across fields and attaches an
``ngio.FeatureTable`` referencing the corresponding ``Label``.
"""
from typing import Dict, Optional
import logging

from ngio import FeatureTable, OmeZarrContainer
import pandas as pd

from blimp.ome_ngff.labels import (
    global_id,
    MAX_OBJECTS_PER_FIELD,
    _validate_field_offset_capacity,
)

logger = logging.getLogger(__name__)


def _offset_feature_table_ids(
    df: pd.DataFrame,
    field_id: int,
    well_name: str,
    well_offset: int,
    max_objects_per_field: int = MAX_OBJECTS_PER_FIELD,
) -> pd.DataFrame:
    """Apply the same ``global_id_numeric = well_offset + field_id *
    max_objects_per_field + local_id`` formula used for label placement to
    a feature dataframe's ``label`` column, and its ``parent_label`` column
    if present (the same field, so the same offset).

    Also adds:
    - ``global_id`` (text, e.g. ``"C09_0004_0000123"``, see
      :func:`blimp.ome_ngff.labels.global_id`) computed from each row's
      *own* (pre-offset) local id -- a human-readable companion to the
      numeric ``label``, which as an integer pixel value has no way to
      carry the well name.
    - ``global_id_numeric`` (``int64``), plate-wide unique -- ``label``
      (after its own per-field offset) plus ``well_offset``, matching the
      plate-level Labels layer's own pixel values
      (:func:`blimp.ome_ngff.plate.build_plate_pyramid`).

    Parameters
    ----------
    df
        One field's own ``quantify()`` output (or a row subset of it),
        with a ``label`` column of locally-unique IDs.
    field_id
        The field's stable, source-assigned identifier (see
        ``FieldLayout.field_ids``).
    well_name
        e.g. ``"C09"`` -- used only to build ``global_id``, not the
        numeric ``label``/``parent_label`` offset itself.
    well_offset
        This well's own plate-wide offset (see
        :func:`blimp.ome_ngff.labels.well_label_offset`), added on top of
        the existing per-field offset to build ``global_id_numeric``.
    max_objects_per_field
        Must match the value used when placing the corresponding ``Label``
        array (:func:`blimp.ome_ngff.labels._offset_label_ids`).

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``label``/``parent_label`` shifted into this
        field's global ID range, and new ``global_id``/``global_id_numeric``
        columns.
    """
    _validate_field_offset_capacity(field_id, max_objects_per_field)
    df = df.copy()
    df["global_id"] = [global_id(well_name, field_id, int(local_id)) for local_id in df["label"]]
    offset = field_id * max_objects_per_field
    df["label"] = df["label"] + offset
    if "parent_label" in df.columns:
        df["parent_label"] = df["parent_label"] + offset
    df["global_id_numeric"] = (df["label"] + well_offset).astype("int64")
    return df


def _write_well_features(
    container: OmeZarrContainer,
    label_name: str,
    field_dataframes: Dict[int, Optional[pd.DataFrame]],
    well_name: str,
    well_offset: int,
    table_name: Optional[str] = None,
    max_objects_per_field: int = MAX_OBJECTS_PER_FIELD,
) -> None:
    """Offset, concatenate, and attach one label's per-field measurements as
    a single ``FeatureTable``.

    A field with no measurements (``None``, e.g. quantification failed or
    was skipped for that field) simply contributes no rows -- its label's
    real objects, if the label itself exists, just have no measurements.

    Parameters
    ----------
    container
        The well's container -- see :func:`blimp.ome_ngff.plate._write_well_image`.
    label_name
        Name of the ``Label`` these measurements reference (the object's
        own name for a non-aggregated ``quantify()`` result, or the
        *parent's* name when ``quantify(aggregate=True)`` was used; the
        caller decides this by which name it passes here, not this
        function).
    field_dataframes
        Maps each field's ``field_id`` (matching ``FieldLayout.field_ids``)
        to its own ``quantify()`` output for this object, or ``None`` if
        missing.
    well_name
        e.g. ``"C09"`` -- see :func:`_offset_feature_table_ids`.
    well_offset
        See :func:`_offset_feature_table_ids`.
    table_name
        Table name (default: ``f"{label_name}_features"``).
    max_objects_per_field
        See :func:`_offset_feature_table_ids`.
    """
    offset_dfs = [
        _offset_feature_table_ids(df, field_id, well_name, well_offset, max_objects_per_field)
        for field_id, df in field_dataframes.items()
        if df is not None
    ]
    if not offset_dfs:
        logger.warning(f"No feature data found for label '{label_name}'; skipping FeatureTable attachment.")
        return

    combined = pd.concat(offset_dfs, ignore_index=True)
    table = FeatureTable(table_data=combined, reference_label=label_name)
    container.add_table(table_name or f"{label_name}_features", table, overwrite=True)
