"""napari viewer utilities for visually inspecting blimp-written OME-NGFF
images. Only depends on ``napari`` and ``ngio``, so it works from a
lightweight viewing environment without blimp's full (much heavier)
conversion dependency set installed.
"""
from typing import Dict, List, Union, Literal, Callable, Optional
from pathlib import Path
import types

from ngio import open_ome_zarr_plate, open_ome_zarr_container
from napari.utils.colormaps import ensure_colormap, DirectLabelColormap
import numpy as np
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


def add_plate(
    viewer: napari.Viewer,
    plate_path: Union[str, Path],
    kind: Literal["stack", "mip"] = "mip",
) -> List["napari.layers.Layer"]:
    """Add every populated well of a whole plate to the viewer at once, laid
    out at its true row/column position -- a fast, full-resolution
    alternative to opening the plate through the ``napari-ome-zarr`` plugin
    directly (that plugin's own plate-stitching eagerly allocates a real
    zero array for every *declared* grid position, at every pyramid level
    and channel, which can take minutes; see ``build_plate_pyramid``).

    Adds one multiscale ``Image`` layer per channel, one multiscale
    ``Labels`` layer per label found on any well (measurements are *not*
    merged/attached here -- a real, measured request cost over a remote
    store for every label found, paid even for labels no one ever
    inspects; call :func:`attach_plate_wide_measurements` afterward for
    the ones you actually want hover-inspectable, keyed to match this
    layer's own plate-wide-unique pixel values), one combined plate-wide
    ``Shapes`` layer outlining each well's own outer boundary (visible by
    default -- this is the useful overview at plate scale), and one
    combined plate-wide ``Shapes`` layer outlining every well's own FOV
    boundaries (each well's ``FOV_ROI_table`` rectangles, shifted to that
    well's grid position; hidden by default -- only useful once zoomed
    into a single well). Point-object tables are not included here -- that
    level of per-object detail belongs in the per-well view
    (``add_points_with_measurements``).

    Every layer here is in plain pixel units (unlike ``add_rois`` and
    friends, there's no real-world-scaled layer from ``napari-ome-zarr`` in
    this viewer to match) -- toggling a label on/off is just napari's own
    layer-list visibility control, nothing this function needs to manage.

    Parameters
    ----------
    viewer
        The napari viewer to add the layers to.
    plate_path
        Full path to the plate's .zarr store.
    kind
        Which image to show -- "stack" or "mip".

    Returns
    -------
    List[napari.layers.Layer]
    """
    # Deferred import: blimp.ome_ngff.plate pulls in blimp's full (much
    # heavier) conversion dependency set (bioio, etc.), which this module's
    # own docstring promises not to require just to import it.
    from blimp.ome_ngff.plate import build_plate_pyramid

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    well_paths = plate.wells_paths()
    if not well_paths:
        raise ValueError(f"Plate {plate_path} has no wells written yet.")

    well_containers = {}
    label_names = set()
    for well_path in well_paths:
        row, column = well_path.split("/")
        # Plain string join, not pathlib -- Path() silently collapses a URL's
        # "http://host/..." into "http:/host/..." (single slash) on any /-join,
        # which breaks open_ome_zarr_container for a remote (http://) plate_path.
        container = open_ome_zarr_container(f"{plate_path}/{well_path}/{kind}")
        well_containers[well_path] = container
        label_names.update(container.list_labels())

    reference_container = next(iter(well_containers.values()))
    channel_names = reference_container.channel_labels
    channel_colors = [
        f"#{channel.channel_visualisation.color}" for channel in reference_container.meta.channels_meta.channels
    ]

    pyramid = build_plate_pyramid(plate_path, kind=kind, plate=plate, open_containers=well_containers)
    layers = list(
        viewer.add_image(
            pyramid,
            multiscale=True,
            channel_axis=0,
            name=channel_names,
            colormap=channel_colors,
            blending="additive",
        )
    )

    n_rows, n_cols = len(plate.rows), len(plate.columns)
    pitch_h = pyramid[0].shape[-2] // n_rows
    pitch_w = pyramid[0].shape[-1] // n_cols

    for label_name in sorted(label_names):
        label_pyramid = build_plate_pyramid(
            plate_path, kind=kind, label_name=label_name, plate=plate, open_containers=well_containers
        )
        # visible=False: a plate-scale label layer is expensive to render and
        # rarely wanted immediately on load -- toggle it on from the layer
        # list (napari's own visibility control) once you actually need it.
        # Its measurements aren't merged/attached here either, for the same
        # reason -- a real, measured cost (over a remote store) for labels no
        # one inspects; call attach_plate_wide_measurements(plate_path,
        # label_name) afterward for the ones you actually want.
        label_layer = viewer.add_labels(label_pyramid, multiscale=True, name=label_name, visible=False)
        layers.append(label_layer)

    fov_rectangles = []
    fov_names = []
    well_rectangles = []
    well_names = []
    for well_path, container in well_containers.items():
        if "FOV_ROI_table" not in container.list_tables():
            continue
        row, column = well_path.split("/")
        y_offset = plate.rows.index(row) * pitch_h
        x_offset = plate.columns.index(column) * pitch_w
        table = container.get_table("FOV_ROI_table")
        pixel_size = container.get_image().pixel_size

        well_y0, well_x0 = float("inf"), float("inf")
        well_y1, well_x1 = float("-inf"), float("-inf")
        for roi in table.rois():
            slices = roi.to_slicing_dict(pixel_size=pixel_size)
            y0, y1 = slices["y"].start + y_offset, slices["y"].stop + y_offset
            x0, x1 = slices["x"].start + x_offset, slices["x"].stop + x_offset
            fov_rectangles.append([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
            fov_names.append(roi.name)
            well_y0, well_x0 = min(well_y0, y0), min(well_x0, x0)
            well_y1, well_x1 = max(well_y1, y1), max(well_x1, x1)

        well_rectangles.append([[well_y0, well_x0], [well_y0, well_x1], [well_y1, well_x1], [well_y1, well_x0]])
        well_names.append(row + column)

    if well_rectangles:
        # visible=True: unlike the per-FOV boundaries below, a plate-scale
        # view is exactly where you want to see which wells are which
        # straight away -- FOV-level detail only becomes useful once zoomed
        # into a single well.
        layers.append(
            viewer.add_shapes(
                well_rectangles,
                shape_type="rectangle",
                name="Well_ROI_table",
                edge_color="cyan",
                face_color="transparent",
                edge_width=8,
                text={"string": well_names, "color": "cyan", "anchor": "center", "size": 14},
                visible=True,
            )
        )

    if fov_rectangles:
        layers.append(
            viewer.add_shapes(
                fov_rectangles,
                shape_type="rectangle",
                name="FOV_ROI_table",
                edge_color="yellow",
                face_color="transparent",
                edge_width=8,
                text={"string": fov_names, "color": "yellow", "anchor": "center", "size": 14},
                visible=False,
            )
        )

    return layers


def attach_plate_wide_measurements(
    viewer: napari.Viewer,
    plate_path: Union[str, Path],
    label_name: str,
    kind: Literal["stack", "mip"] = "mip",
) -> "napari.layers.Labels":
    """Attach ``label_name``'s plate-wide feature table (every contributing
    well's own measurements, merged -- see
    ``blimp.ome_ngff.plate._read_plate_wide_features``) onto the ``Labels``
    layer :func:`add_plate` already added for it, on demand.

    ``add_plate`` itself no longer does this eagerly for every label found
    -- a real, measured request cost (over a remote store) for labels no
    one ever inspects. Call this afterward for whichever label(s) you
    actually want hover-inspectable (e.g. via a tool like Napari Feature
    Visualizer) -- for coloring by one specific feature instead, see
    :func:`add_feature_heatmap`, which is unaffected by this and already
    loads independently, on demand.

    Parameters
    ----------
    viewer
        The napari viewer ``add_plate`` was called on.
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to attach -- must match the name of
        a ``Labels`` layer ``add_plate`` already added.
    kind
        "stack" or "mip".

    Returns
    -------
    napari.layers.Labels
        The same layer, with ``.features`` now set.

    Raises
    ------
    LookupError
        If no ``Labels`` layer named ``label_name`` exists in
        ``viewer.layers`` (call ``add_plate`` first, or check the name
        matches).
    ValueError
        If no well has a matching features table.
    """
    try:
        layer = next(
            layer for layer in viewer.layers if layer.name == label_name and isinstance(layer, napari.layers.Labels)
        )
    except StopIteration:
        raise LookupError(f"No Labels layer named {label_name!r} found -- call add_plate first.") from None

    # Deferred import: see add_plate's own comment above.
    from blimp.ome_ngff.plate import _read_plate_wide_features

    features_df = _read_plate_wide_features(plate_path, label_name, kind=kind)
    if features_df is None:
        raise ValueError(f"No well in {plate_path} has a {label_name}_features table.")

    layer.features = features_df
    return layer


def add_feature_heatmap(
    viewer: napari.Viewer,
    plate_path: Union[str, Path],
    label_name: str,
    feature_name: str,
    kind: Literal["stack", "mip"] = "mip",
    colormap: str = "magma",
) -> "napari.layers.Labels":
    """Add a plate-wide feature heatmap: a new ``Labels`` layer colored by
    each object's own ``feature_name`` measurement instead of by label
    identity.

    Built with a ``DirectLabelColormap`` -- a dict from each object's own
    plate-wide-unique pixel value (``global_id_numeric``) to a fixed RGBA
    color, computed once from the merged features table via ``colormap``
    and this heatmap's own contrast limits (1st/99th percentile of
    ``feature_name``). This is deliberately *not* what a tool like Napari
    Feature Visualizer does -- its own colormap computation builds a dense
    array sized to the largest label ID present, which crashes at plate
    scale since plate-wide IDs can run into the trillions (see
    ``blimp.ome_ngff.labels.well_label_offset``). ``DirectLabelColormap``
    is dict-based instead: napari remaps raw label values through a
    compact GPU texture built from the dict's own keys, so cost scales
    with the number of *objects*, not the ID magnitude, and only touches
    whatever's actually visible at the current zoom -- a full 384-well
    plate costs the same as a two-well one.

    Any pixel with no entry in the dict -- background (label 0), or an
    object with no matching feature row -- renders fully transparent, via
    the same code path every ordinary ``Labels`` layer already uses for
    its own background, revealing whatever's drawn underneath.

    A genuinely new layer, added alongside (not replacing) any plain
    ``Labels`` layer ``add_plate`` may have already added for
    ``label_name``.

    Parameters
    ----------
    viewer
        The napari viewer to add the layer to.
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to show (``f"{label_name}_features"``).
    feature_name
        Which column of that features table to show.
    kind
        "stack" or "mip".
    colormap
        Any napari/vispy colormap name.

    Returns
    -------
    napari.layers.Labels
    """
    # Deferred import: see add_plate's own comment above.
    from blimp.ome_ngff.plate import (
        build_plate_pyramid,
        _read_plate_wide_features,
        _read_plate_wide_feature_raw,
    )

    # _read_plate_wide_feature_raw reads only this one column (plus object ids)
    # directly via zarr instead of ngio.FeatureTable.dataframe's whole-table read
    # -- a real, measured ~4.5x fewer requests for this specific case. It can't
    # distinguish "no table at all" from "table exists but lacks this column" on
    # its own, so on failure fall back to a full read just to build a precise
    # error message (an accepted, occasional cost -- see _read_plate_wide_features's
    # own docstring on discovering available feature names).
    features_df = _read_plate_wide_feature_raw(plate_path, label_name, feature_name, kind=kind)
    if features_df is None:
        full_df = _read_plate_wide_features(plate_path, label_name, kind=kind)
        if full_df is None:
            raise ValueError(f"No well in {plate_path} has a {label_name}_features table.")
        raise ValueError(
            f"{feature_name!r} is not a column of {label_name}_features; available: {sorted(full_df.columns)}"
        )

    values = features_df[feature_name].to_numpy(dtype=float)
    clim_low, clim_high = np.nanpercentile(values, [1, 99])
    normalized = np.clip((values - clim_low) / (clim_high - clim_low), 0.0, 1.0)
    colors = ensure_colormap(colormap).map(normalized)

    color_dict = dict(zip(features_df["label"].tolist(), colors))
    color_dict[None] = "transparent"

    label_pyramid = build_plate_pyramid(plate_path, kind=kind, label_name=label_name)
    return viewer.add_labels(
        label_pyramid,
        multiscale=True,
        name=f"{label_name}: {feature_name}",
        colormap=DirectLabelColormap(color_dict=color_dict),
    )


# Functions bound onto a viewer instance by add_blimp_napari_methods(), keyed
# by the method name they become. Add a new entry here to make a new
# function available as viewer.<name>(...) too.
_VIEWER_METHODS: Dict[str, Callable] = {
    "add_rois": add_rois,
    "add_labels_with_measurements": add_labels_with_measurements,
    "add_points_with_measurements": add_points_with_measurements,
    "add_plate": add_plate,
    "attach_plate_wide_measurements": attach_plate_wide_measurements,
    "add_feature_heatmap": add_feature_heatmap,
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
