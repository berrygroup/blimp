"""napari viewer utilities for visually inspecting blimp-written OME-NGFF images.

Standalone from the rest of blimp's conversion/processing pipeline -- only
depends on ``napari`` and ``ngio``, so it works from a lightweight interactive
viewing environment that doesn't have blimp's full (much heavier) dependency
set installed, e.g. via ``pip install -e /path/to/blimp --no-deps``.
"""
from typing import Dict, List, Union, Callable, Optional
from pathlib import Path
import types

from ngio import open_ome_zarr_container
import napari


def add_rois(
    viewer: napari.Viewer,
    image_group_path: Union[str, Path],
    table_name: str = "FOV_ROI_table",
    name: Optional[str] = None,
    edge_color: str = "yellow",
    edge_width: int = 8,
    show_labels: bool = True,
) -> "napari.layers.Shapes":
    """Overlay an ``ngio`` ROI table's regions as rectangles on a viewer.

    Reads the named table via ``ngio``'s own read API (it's stored as an
    AnnData-backed zarr group, not something worth reimplementing by hand)
    and converts each ROI's world-coordinate bounds to the pixel-space
    rectangle corners napari's ``Shapes`` layer expects.

    Parameters
    ----------
    viewer
        The napari viewer to add the layer to.
    image_group_path
        Path to the OME-Zarr image group the table is attached to, e.g.
        ``plate.zarr/C/09/mip``.
    table_name
        Name of the ROI table to read.
    name
        Shapes layer name (default: ``table_name``).
    edge_color, edge_width
        Passed through to ``viewer.add_shapes``.
    show_labels
        Whether to label each rectangle with its ROI name.

    Returns
    -------
    napari.layers.Shapes
    """
    container = open_ome_zarr_container(str(image_group_path))
    table = container.get_table(table_name)
    pixel_size = container.get_image().pixel_size

    rectangles = []
    names = []
    for roi in table.rois():
        slices = roi.to_slicing_dict(pixel_size=pixel_size)
        y0, y1 = slices["y"].start, slices["y"].stop
        x0, x1 = slices["x"].start, slices["x"].stop
        rectangles.append([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        names.append(roi.name)

    kwargs = {}
    if show_labels:
        kwargs["text"] = {"string": names, "color": edge_color, "anchor": "upper_left", "size": 14}

    return viewer.add_shapes(
        rectangles,
        shape_type="rectangle",
        name=name or table_name,
        edge_color=edge_color,
        face_color="transparent",
        edge_width=edge_width,
        # Rectangle corners above are in pixel units (to_slicing_dict converted
        # them); without this, napari treats those pixel values as world units
        # directly (implicit scale=1), while the image/label layers opened via
        # the napari-ome-zarr plugin already carry the real physical scale from
        # NGFF metadata -- the two would disagree on what one world unit means,
        # so this layer would render at the wrong size/position relative to them.
        scale=(pixel_size.y, pixel_size.x),
        **kwargs,
    )


def add_labels_with_measurements(
    viewer: napari.Viewer,
    image_group_path: Union[str, Path],
    label_name: str,
    feature_table_name: Optional[str] = None,
) -> "napari.layers.Labels":
    """Add an ``ngio`` ``Label`` (+ its ``FeatureTable``, if present) as one
    napari Labels layer with linked per-object features.

    The ``add_rois`` equivalent for labels: read via ``ngio``'s own read
    API, convert to napari's native representation, add to the viewer,
    return the layer. Not named ``add_labels`` -- ``napari.Viewer`` already
    has a built-in method of that name, and a same-named function here
    would be confusing to call alongside it.

    Parameters
    ----------
    viewer
        The napari viewer to add the layer to.
    image_group_path
        Path to the OME-Zarr image group the label is attached to, e.g.
        ``plate.zarr/C/09/mip``.
    label_name
        Name of the label to read (e.g. ``"Nuclei"``).
    feature_table_name
        Name of the feature table to read (default:
        ``f"{label_name}_features"``, the naming convention
        ``blimp.ome_ngff.features._write_well_features`` uses). Pass ``""``
        explicitly to add the label with no features.

    Returns
    -------
    napari.layers.Labels
    """
    container = open_ome_zarr_container(str(image_group_path))
    label = container.get_label(label_name)
    data = label.get_as_numpy()

    features = None
    table_name = feature_table_name if feature_table_name is not None else f"{label_name}_features"
    if table_name and table_name in container.list_tables():
        # ngio's FeatureTable.dataframe carries "label" as the index, but
        # napari's add_labels(features=...) matches rows to label values by
        # column, not index -- an indexed DataFrame is silently accepted but
        # matched by row position instead, so the index needs converting
        # back to a plain "label" column first.
        features = container.get_feature_table(table_name).dataframe.reset_index()

    # Without this, napari displays `data` in raw pixel units (implicit
    # scale=1) -- disagreeing with any image/labels layer opened via the
    # napari-ome-zarr plugin, which already carries the real physical scale
    # from NGFF metadata. (t, z, y, x) trimmed to data's own leading axes.
    pixel_size = label.pixel_size
    scale = (pixel_size.t, pixel_size.z, pixel_size.y, pixel_size.x)[-data.ndim :]

    return viewer.add_labels(data, name=label_name, features=features, scale=scale)


def add_points_with_measurements(
    viewer: napari.Viewer,
    image_group_path: Union[str, Path],
    table_name: str,
    name: Optional[str] = None,
    size: float = 10,
    face_color: str = "yellow",
) -> "napari.layers.Points":
    """Add a point-object ``GenericRoiTable`` (written by
    ``blimp.ome_ngff.labels._write_well_points``) as a napari Points layer
    with linked per-point features.

    Point objects have no pixel-value identity to read back as a Labels
    layer (see ``_write_well_points``) -- each is instead one small ROI in a
    ``GenericRoiTable``, read here via ``ngio``'s own read API and converted
    to one napari point per ROI, at its own center coordinate.

    Parameters
    ----------
    viewer
        The napari viewer to add the layer to.
    image_group_path
        Path to the OME-Zarr image group the table is attached to, e.g.
        ``plate.zarr/C/09/mip``.
    table_name
        Name of the point-object table to read (the ``channel_name`` given
        to ``_write_well_points``, e.g. ``"Spots"``).
    name
        Points layer name (default: ``table_name``).
    size
        Point marker diameter, in pixels.
    face_color
        Passed through to ``viewer.add_points``.

    Returns
    -------
    napari.layers.Points
    """
    container = open_ome_zarr_container(str(image_group_path))
    table = container.get_table(table_name)
    pixel_size = container.get_image().pixel_size

    coords = []
    roi_names = []
    axes: List[str] = []
    for roi in table.rois():
        slices = roi.to_slicing_dict(pixel_size=pixel_size)
        axes = [axis for axis in ("z", "y", "x") if axis in slices]
        point = [(slices[axis].start + slices[axis].stop) / 2 for axis in axes]
        coords.append(point)
        roi_names.append(roi.name)

    features = None
    df = table.dataframe
    geometry_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    measurement_columns = [c for c in df.columns if c not in geometry_columns]
    if measurement_columns:
        features = df.loc[roi_names, measurement_columns].reset_index(drop=True)

    # coords above are in pixel units (to_slicing_dict converted them); without
    # a matching scale, napari would treat those pixel values as world units
    # directly, disagreeing with any image/labels layer opened via the
    # napari-ome-zarr plugin, which already carries the real physical scale
    # from NGFF metadata.
    axis_to_pixel_size = {"z": pixel_size.z, "y": pixel_size.y, "x": pixel_size.x}
    scale = tuple(axis_to_pixel_size[axis] for axis in axes)

    return viewer.add_points(
        coords, name=name or table_name, size=size, face_color=face_color, features=features, scale=scale
    )


# Functions bound onto a viewer instance by add_blimp_napari_methods(), keyed
# by the method name they become. Add a new entry here to make a new
# function available as viewer.<name>(...) too.
_VIEWER_METHODS: Dict[str, Callable] = {
    "add_rois": add_rois,
    "add_labels_with_measurements": add_labels_with_measurements,
    "add_points_with_measurements": add_points_with_measurements,
}


def add_blimp_napari_methods(viewer: napari.Viewer) -> napari.Viewer:
    """Bind this module's viewer-utility functions onto one viewer instance.

    Purely a convenience for interactive use -- lets you call e.g.
    ``viewer.add_rois(image_group_path)`` instead of
    ``add_rois(viewer, image_group_path)``. Implemented with
    ``types.MethodType``, which binds a plain function to one specific
    instance rather than patching the ``napari.Viewer`` class itself, so
    only viewers passed through this function gain the extra methods --
    other viewers, and other code importing napari elsewhere in the same
    process, are unaffected.

    ``napari.Viewer`` is a pydantic model with ``validate_assignment=True``,
    so a plain ``setattr`` validates the new attribute against the model's
    declared fields and raises ``ValidationError`` for anything not already
    part of its schema -- which a bound method never is.
    ``object.__setattr__`` bypasses that validation layer and writes
    directly to the instance's own ``__dict__``.

    Parameters
    ----------
    viewer
        The napari viewer instance to extend in place.

    Returns
    -------
    napari.Viewer
        The same viewer, so this can be chained, e.g.
        ``viewer = add_blimp_napari_methods(napari.Viewer())``.
    """
    for method_name, func in _VIEWER_METHODS.items():
        object.__setattr__(viewer, method_name, types.MethodType(func, viewer))
    return viewer
