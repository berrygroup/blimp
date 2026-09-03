"""napari viewer utilities for visually inspecting blimp-written OME-NGFF images.

Standalone from the rest of blimp's conversion/processing pipeline -- only
depends on ``napari`` and ``ngio``, so it works from a lightweight interactive
viewing environment that doesn't have blimp's full (much heavier) dependency
set installed, e.g. via ``pip install -e /path/to/blimp --no-deps``.
"""
from typing import Dict, Union, Callable, Optional
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
        **kwargs,
    )


# Functions bound onto a viewer instance by add_blimp_napari_methods(), keyed
# by the method name they become. Add a new entry here to make a new
# function available as viewer.<name>(...) too.
_VIEWER_METHODS: Dict[str, Callable] = {
    "add_rois": add_rois,
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

    ``napari.Viewer`` is a pydantic model with ``validate_assignment=True``
    (confirmed against a real instance's ``model_config``), so a plain
    ``setattr`` validates the new attribute against the model's declared
    fields and raises ``ValidationError`` for anything not already part of
    its schema -- which a bound method never is. ``object.__setattr__``
    bypasses that validation layer and writes directly to the instance's
    own ``__dict__``, which is what actually needs to happen here.

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
