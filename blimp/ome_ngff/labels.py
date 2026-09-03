"""Segmentation label handling for OME-NGFF assembly.

Two fundamentally different shapes, per object type (see
``quantify.py``'s own two code paths):

- **Blob objects** (regular segmented objects) have a stable "pixel value =
  object identity" -- placed into a well-scale ``ngio.Label`` array, with
  each field's locally-unique IDs shifted into a disjoint, reproducible
  global range first (:func:`_offset_label_ids`).
- **Point objects** have no such stable pixel-value identity (``quantify()``
  only takes a binary mask and assigns an arbitrary scan-order index) --
  stored instead as one ``ngio.GenericRoiTable`` per channel, one small ROI
  per point.
"""
from typing import Dict, Optional
import logging

from ngio import Roi, GenericRoiTable, OmeZarrContainer
import numpy as np
import pandas as pd

from blimp.ome_ngff.layout import FieldLayout

logger = logging.getLogger(__name__)

# Upper bound on objects per field, used to shift each field's locally-unique
# label IDs into a disjoint global range: global_id = field_id *
# MAX_OBJECTS_PER_FIELD + local_id. 10 million comfortably supports
# realistic field counts within uint32's ~4.29e9 ceiling (~400 fields/well
# headroom) while keeping the paired human-readable id
# (f"{well}_{field_id:04d}_{local_id:07d}") consistent with this same bound.
MAX_OBJECTS_PER_FIELD = 10_000_000


def _offset_label_ids(
    local_array: np.ndarray, field_id: int, max_objects_per_field: int = MAX_OBJECTS_PER_FIELD
) -> np.ndarray:
    """Shift one field's locally-unique label IDs into a disjoint,
    reproducible global range.

    ``global_id = field_id * max_objects_per_field + local_id``, keyed only
    on the stable ``field_id`` (not an arbitrary running count across
    however many fields happen to be processed, in whatever order) -- so
    the same field always gets the same global IDs regardless of processing
    order, and dividing a global ID back by ``max_objects_per_field``
    recovers its source ``field_id``.

    Parameters
    ----------
    local_array
        One field's own label array, with locally-unique integer IDs
        starting at 1 (0 = background).
    field_id
        The field's stable, source-assigned identifier (see
        ``FieldLayout.field_ids``).
    max_objects_per_field
        Upper bound on objects per field -- raises ``ValueError`` if
        exceeded rather than silently colliding with the next field's ID
        block.

    Returns
    -------
    numpy.ndarray
        Same shape as ``local_array`` (background stays 0), with every
        nonzero pixel shifted into its field's own global ID range.
    """
    local_max = int(local_array.max()) if local_array.size else 0
    if local_max >= max_objects_per_field:
        raise ValueError(
            f"Field {field_id} has a local object ID ({local_max}) that meets or exceeds "
            f"MAX_OBJECTS_PER_FIELD ({max_objects_per_field}) -- global IDs would collide "
            "with the next field's range. Increase MAX_OBJECTS_PER_FIELD."
        )
    offset = field_id * max_objects_per_field
    return np.where(local_array > 0, local_array.astype(np.int64) + offset, 0).astype(np.uint32)


def fov_object_id(well_name: str, field_id: int, local_id: int) -> str:
    """Human-readable, traceable object identifier, e.g. ``"C09_0004_0000123"``
    -- derived from the same two numbers as the numeric ``global_id``
    (:func:`_offset_label_ids`), so the two never disagree."""
    return f"{well_name}_{field_id:04d}_{local_id:07d}"


def _write_well_labels(
    container: OmeZarrContainer,
    layout: FieldLayout,
    label_name: str,
    field_arrays: Dict[int, Optional[np.ndarray]],
    max_objects_per_field: int = MAX_OBJECTS_PER_FIELD,
) -> None:
    """Stitch one named label channel's per-field arrays into one
    well-scale ``Label`` with reproducible global IDs, and attach its
    masking ROI table (one bounding-box ROI per object, computed
    automatically by ``ngio`` directly from the assembled label array).

    Parameters
    ----------
    container
        The well's already-written intensity image container (see
        :func:`blimp.ome_ngff.plate._write_well_image`) -- the new label is
        derived from its multiscale pyramid, so it gets a matching one.
    layout
        As returned by ``get_field_layout_from_tiff_metadata``.
    label_name
        Name for the new label (e.g. ``"Nuclei"``).
    field_arrays
        Maps each field's ``field_id`` (matching ``layout.field_ids``) to
        its own local 2D (Y, X) label array with locally-unique integer IDs
        starting at 1 -- or ``None`` for a field whose label TIFF was
        missing, substituted with a blank (all-zero) region.
    max_objects_per_field
        See :func:`_offset_label_ids`.
    """
    label = container.derive_label(label_name, ref_image=container.get_image(), overwrite=True)

    canvas = np.zeros(label.shape, dtype=np.uint32)
    h, w = layout.tile_shape[3], layout.tile_shape[4]

    for field_index, (y0, x0) in enumerate(layout.offsets):
        field_id = layout.field_ids[field_index]
        local_array = field_arrays.get(field_id)
        if local_array is None:
            continue  # blank region; canvas is already zero there
        canvas[..., y0 : y0 + h, x0 : x0 + w] = _offset_label_ids(local_array, field_id, max_objects_per_field)

    label.set_array(canvas, merge="keep_nonzero")

    roi_table = container.build_masking_roi_table(label_name)
    container.add_table(f"{label_name}_ROI_table", roi_table, overwrite=True)


def _write_well_points(
    container: OmeZarrContainer,
    layout: FieldLayout,
    well_name: str,
    channel_name: str,
    field_masks: Dict[int, Optional[np.ndarray]],
    field_features: Dict[int, Optional[pd.DataFrame]],
    max_objects_per_field: int = MAX_OBJECTS_PER_FIELD,
) -> None:
    """Attach one point-object channel's per-field points as a single
    ``GenericRoiTable`` (one small ROI per point, at its own world
    coordinate) plus its measurement columns.

    Point objects have no meaningful pixel-value identity to place in a
    ``Label`` array -- ``quantify()`` itself only takes a binary mask
    (``label_array > 0``) and assigns each nonzero pixel an arbitrary
    scan-order index via ``np.argwhere``. Recovering each point's own pixel
    coordinate therefore requires re-running that same ``np.argwhere`` over
    the same per-field mask, in the same order, and zipping the result back
    against ``quantify()``'s own sequential ``label`` column (1..N per
    field) -- not by pixel value, which point objects don't have.

    Parameters
    ----------
    container
        The well's already-written intensity image container.
    layout
        As returned by ``get_field_layout_from_tiff_metadata``.
    well_name
        e.g. ``"C09"`` -- used to build each point's human-readable id.
    channel_name
        Name for the new table (e.g. ``"Spots"``).
    field_masks
        Maps each field's ``field_id`` to its own local 2D (Y, X) boolean/
        binary point mask, or ``None`` if missing.
    field_features
        Maps each field's ``field_id`` to its own ``quantify()`` output for
        this channel (sequential ``label`` 1..N, matching ``np.argwhere``
        scan order over that field's own mask), or ``None`` if missing.
    max_objects_per_field
        See :func:`_offset_label_ids`.
    """
    pixel_size_x = layout.pixel_size_x if layout.pixel_size_x and layout.pixel_size_x > 0 else 1.0
    pixel_size_y = layout.pixel_size_y if layout.pixel_size_y and layout.pixel_size_y > 0 else 1.0

    rois = []
    feature_rows = []
    for field_index, (y0, x0) in enumerate(layout.offsets):
        field_id = layout.field_ids[field_index]
        mask = field_masks.get(field_id)
        if mask is None:
            continue

        coords = np.argwhere(mask > 0)
        offset = field_id * max_objects_per_field
        features_df = field_features.get(field_id)

        for local_id_minus_one, (y, x) in enumerate(coords):
            local_id = local_id_minus_one + 1
            global_id = int(offset + local_id)
            name = fov_object_id(well_name, field_id, local_id)
            world_y = float((y0 + y) * pixel_size_y)
            world_x = float((x0 + x) * pixel_size_x)
            rois.append(
                Roi.from_values(
                    slices={
                        "y": slice(world_y, world_y + pixel_size_y),
                        "x": slice(world_x, world_x + pixel_size_x),
                    },
                    name=name,
                    label=global_id,
                    space="world",
                )
            )
            if features_df is not None:
                row = features_df.loc[features_df["label"] == local_id]
                if not row.empty:
                    feature_rows.append({"FieldIndex": name, **row.iloc[0].drop("label").to_dict()})

    table = GenericRoiTable(rois=rois)
    if feature_rows:
        features_df = pd.DataFrame(feature_rows).set_index("FieldIndex")
        merged = table.dataframe.join(features_df)
        table.set_table_data(merged)

    container.add_table(channel_name, table, overwrite=True)
