"""Convert Nikon nd2 files to OME-NGFF (OME-Zarr) format.

Each nd2 file is treated as a single well containing several stage-position
fields of view, which are stitched into one contiguous mosaic image per well
using nominal grid placement from stage coordinates (no sub-pixel
registration). Wells are written into one shared OME-Zarr plate store per
plate (HCS layout), rather than one independent store per well, to avoid the
"many small files" problem that a store-per-well layout would create.

Plate/well registration uses ``ngio`` (pinned to an exact version -- see
``requirements.txt``) for its atomic, file-locked plate/well metadata
updates, which make concurrent per-well writers (e.g. one PBS job per nd2
file) safe. The actual image arrays and NGFF 0.5 metadata are written
directly against ``zarr``, independent of ``ngio``'s own image-writing API.
"""
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from pathlib import Path
import re
import logging

from ngio import (
    Roi,
    RoiTable,
    create_empty_plate,
    NgioFileExistsError,
    open_ome_zarr_plate,
    NgioFileNotFoundError,
    open_ome_zarr_container,
)
from bioio import BioImage
from bioio_ome_zarr import Reader as OmeZarrReader
from ngio.hcs._plate import OmeZarrPlate
import nd2
import zarr
import numpy as np

from blimp.log import configure_logging
from blimp.preprocessing.nd2_to_ome_tiff import _get_list_of_files_current_batch

logger = logging.getLogger(__name__)

NGFF_VERSION = "0.5"
NUM_PYRAMID_LEVELS = 5

# Matches "C09", "AA12", etc. -- one or more letters (row) followed by digits
# (column).
_WELL_NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)")
# Matches a well name embedded in a filename, e.g. "WellC09_Channel...".
_FILENAME_WELL_RE = re.compile(r"Well([A-Za-z]+\d+)")


class FieldLayout:
    """Stage-position-derived layout for stitching an nd2 file's fields of
    view into one mosaic canvas.

    Attributes
    ----------
    row, column
        Well identifier, parsed from the nd2 file's own position-name
        metadata (preferred) or its filename (fallback).
    offsets
        One ``(y0, x0)`` pixel offset per field, in field/position order.
    tile_shape
        ``(T, C, Z, Y, X)`` shape of a single field.
    canvas_shape
        ``(T, C, Z, Y, X)`` shape of the full stitched mosaic.
    pixel_size_x, pixel_size_y, pixel_size_z
        Physical pixel size in micrometers.
    channel_names, channel_colors
        One entry per channel, from the nd2 file's own channel metadata.
    position_names
        One nd2 position ``name`` per field (from ``XYPosLoop``), in the
        same order as ``offsets``. ``None`` for a field with no recorded
        position name (e.g. a single-position file).
    """

    def __init__(
        self,
        row: str,
        column: int,
        offsets: List[Tuple[int, int]],
        tile_shape: Tuple[int, int, int, int, int],
        canvas_shape: Tuple[int, int, int, int, int],
        pixel_size_x: float,
        pixel_size_y: float,
        pixel_size_z: float,
        channel_names: List[str],
        channel_colors: List[str],
        position_names: List[Optional[str]],
    ):
        self.row = row
        self.column = column
        self.offsets = offsets
        self.tile_shape = tile_shape
        self.canvas_shape = canvas_shape
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.pixel_size_z = pixel_size_z
        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.position_names = position_names


def _parse_well_name(nd2_path: Union[str, Path], position_names: List[Optional[str]]) -> Tuple[str, int]:
    """Determine a well's row/column identifier.

    Prefers the nd2 file's own internal position-name metadata (e.g. a
    position named ``"C09_0000"``) over the filename, since it is generated
    by the acquisition software itself rather than a naming convention that
    could be lost on copy/rename.

    Parameters
    ----------
    nd2_path
        Full path to the .nd2 file (used for the filename fallback).
    position_names
        Position ``name`` values from the file's ``XYPosLoop`` metadata,
        one per field of view. May contain ``None`` entries.

    Returns
    -------
    (row, column)
        e.g. ``("C", 9)``.

    Raises
    ------
    ValueError
        If no well identifier can be determined from either source.
    """
    for name in position_names:
        if name is None:
            continue
        match = _WELL_NAME_RE.match(name)
        if match is not None:
            return match.group(1).upper(), int(match.group(2))

    filename_match = _FILENAME_WELL_RE.search(Path(nd2_path).stem)
    if filename_match is not None:
        well_match = _WELL_NAME_RE.match(filename_match.group(1))
        if well_match is not None:
            return well_match.group(1).upper(), int(well_match.group(2))

    raise ValueError(
        f"Could not determine a well identifier for {nd2_path} from its position names or filename. "
        "Expected a position name or filename containing e.g. 'C09'."
    )


def _cluster_grid_index(values: np.ndarray, tile_extent: float) -> np.ndarray:
    """Assign each stage-coordinate value a 0-indexed row/column cluster.

    Fields meant to sit on the same nominal row/column can still differ by a
    small amount (a few percent of the tile extent, confirmed against a real
    acquisition) -- stage positioning jitter, or a slight rotation between
    the stage and camera axes. Using that raw value directly as a continuous
    pixel offset bakes the jitter into the canvas as a small but visible
    misalignment between tiles that are meant to sit flush (fields are
    acquired with no deliberate overlap). Clustering first, then placing
    each cluster at an exact multiple of the tile size (see
    ``get_field_layout``), snaps to the intended grid instead.

    Parameters
    ----------
    values
        Raw stage coordinates along one axis, one per field.
    tile_extent
        Tile size along that axis, in the same physical units as ``values``.

    Returns
    -------
    numpy.ndarray
        0-indexed cluster id per field, in ``values``' original order.
    """
    order = np.argsort(values)
    cluster = np.zeros(len(values), dtype=int)
    next_id = 0
    for i in range(1, len(order)):
        if values[order[i]] - values[order[i - 1]] > tile_extent / 2:
            next_id += 1
        cluster[order[i]] = next_id
    return cluster


def get_field_layout(
    nd2_path: Union[str, Path],
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
) -> FieldLayout:
    """Compute the stitching layout for an nd2 file's fields of view.

    Reads stage positions, voxel size, and channel metadata directly from
    the nd2 file via the ``nd2`` package's own structured API, rather than
    via the older ``nd2reader``-based metadata pipeline in
    ``nd2_parse_metadata.py``, for more robust cross-version compatibility.

    Parameters
    ----------
    nd2_path
        Full path to the .nd2 image file.
    y_direction
        Direction of increasing (stage) y-coordinates (possible values are
        "up" and "down"), matching the convention already established in
        ``nd2_parse_metadata.py::get_standard_field_id_mapping``.
    x_direction
        Direction of increasing (stage) x-coordinates relative to the image
        ("left" or "right"). Unlike ``y_direction``, this has no precedent
        elsewhere in blimp -- ``nd2_to_ome_tiff.py``/``nd2_parse_metadata.py``
        never needed it, since they number fields rather than placing them on
        a stitched canvas. Added after a real microscope was found to need
        it: some stage/camera combinations report x independently of y, so a
        correct y_direction does not guarantee x needs no equivalent flip.
        Defaults to "left" -- confirmed correct (alongside y_direction="down")
        against a real acquisition, and the most commonly used setup in this
        lab. Override per-instrument if a different microscope needs "right"
        instead.
    placement
        "grid" (default) clusters fields into rows/columns first (see
        :func:`_cluster_grid_index`) and places each at an exact multiple of
        the tile size -- confirmed against a real acquisition to look better
        when fields are meant to sit flush with no deliberate overlap, since
        it discards the small (a few percent of tile size) stage jitter/axis
        cross-talk a continuous offset would otherwise bake into the canvas.
        "continuous" uses the raw stage-position offset directly instead,
        which is more faithful to the actual recorded positions -- prefer it
        if fields have genuine deliberate overlap, or nominal positions are
        not expected to fall on a regular grid.

    Returns
    -------
    FieldLayout
    """
    if y_direction not in {"up", "down"}:
        raise ValueError(f'y_direction = {y_direction}, only "up" or "down" are possible')
    if x_direction not in {"left", "right"}:
        raise ValueError(f'x_direction = {x_direction}, only "left" or "right" are possible')
    if placement not in {"grid", "continuous"}:
        raise ValueError(f'placement = {placement}, only "grid" or "continuous" are possible')

    with nd2.ND2File(str(nd2_path)) as f:
        voxel_size = f.voxel_size()
        sizes = f.sizes
        tile_shape = (
            sizes.get("T", 1),
            sizes.get("C", 1),
            sizes.get("Z", 1),
            sizes["Y"],
            sizes["X"],
        )

        positions = None
        for loop in f.experiment:
            if type(loop).__name__ == "XYPosLoop":
                positions = loop.parameters.points
                break

        if positions is None:
            # single-position acquisition: one field at a nominal (0, 0) offset
            position_names: List[Optional[str]] = [None]
            offsets = [(0, 0)]
        else:
            position_names = [p.name for p in positions]
            stage_x = np.array([p.stagePositionUm.x for p in positions])
            stage_y = np.array([p.stagePositionUm.y for p in positions])

            if placement == "grid":
                tile_extent_x = tile_shape[4] * voxel_size.x
                tile_extent_y = tile_shape[3] * voxel_size.y
                col_idx = _cluster_grid_index(stage_x, tile_extent_x)
                row_idx = _cluster_grid_index(stage_y, tile_extent_y)
                if x_direction == "left":
                    col_idx = col_idx.max() - col_idx
                if y_direction == "up":
                    row_idx = row_idx.max() - row_idx
                offset_x_px = col_idx * tile_shape[4]
                offset_y_px = row_idx * tile_shape[3]
            else:
                if x_direction == "right":
                    offset_x_px = np.round((stage_x - stage_x.min()) / voxel_size.x).astype(int)
                else:
                    offset_x_px = np.round((stage_x.max() - stage_x) / voxel_size.x).astype(int)
                if y_direction == "down":
                    offset_y_px = np.round((stage_y - stage_y.min()) / voxel_size.y).astype(int)
                else:
                    offset_y_px = np.round((stage_y.max() - stage_y) / voxel_size.y).astype(int)
            offsets = list(zip(offset_y_px.tolist(), offset_x_px.tolist()))

        row, column = _parse_well_name(nd2_path, position_names)

        canvas_y = max(y0 for y0, _ in offsets) + tile_shape[3]
        canvas_x = max(x0 for _, x0 in offsets) + tile_shape[4]
        canvas_shape = (tile_shape[0], tile_shape[1], tile_shape[2], canvas_y, canvas_x)

        channel_names = [ch.channel.name for ch in f.metadata.channels]
        channel_colors = [
            f"{ch.channel.color.r:02X}{ch.channel.color.g:02X}{ch.channel.color.b:02X}" for ch in f.metadata.channels
        ]

    return FieldLayout(
        row=row,
        column=column,
        offsets=offsets,
        tile_shape=tile_shape,
        canvas_shape=canvas_shape,
        pixel_size_x=voxel_size.x,
        pixel_size_y=voxel_size.y,
        pixel_size_z=voxel_size.z,
        channel_names=channel_names,
        channel_colors=channel_colors,
        position_names=position_names,
    )


def _build_ngff_v05_metadata(
    image_name: str,
    num_levels: int,
    pixel_size_x: float,
    pixel_size_y: float,
    pixel_size_z: float,
    channel_names: List[str],
    channel_colors: List[str],
) -> Dict[str, Any]:
    """Build NGFF 0.5 multiscale/omero metadata for a TCZYX image group.

    Constructed directly from the public NGFF 0.5 specification
    (https://ngff.openmicroscopy.org/0.5/), not copied from any existing
    writer implementation.

    Parameters
    ----------
    image_name
        Name recorded in the multiscales metadata.
    num_levels
        Number of pyramid levels (level 0 is full resolution; each
        subsequent level halves Y and X).
    pixel_size_x, pixel_size_y, pixel_size_z
        Physical pixel size in micrometers, at full resolution.
    channel_names, channel_colors
        One entry per channel; colors as 6-character hex strings.

    Returns
    -------
    dict
        The ``attributes`` dict to attach to the image's Zarr group.
    """
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = []
    for level in range(num_levels):
        factor = 2**level
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, pixel_size_z, pixel_size_y * factor, pixel_size_x * factor],
                    }
                ],
            }
        )

    omero_channels = [
        {
            "label": name,
            "color": color,
            "active": True,
            "window": {"start": 0, "end": 65535, "min": 0, "max": 65535},
        }
        for name, color in zip(channel_names, channel_colors)
    ]

    return {
        "ome": {
            "version": NGFF_VERSION,
            "multiscales": [
                {
                    "name": image_name,
                    "axes": axes,
                    "datasets": datasets,
                }
            ],
            "omero": {"channels": omero_channels},
        }
    }


def _downsample_yx(array: np.ndarray) -> np.ndarray:
    """Downsample a TCZYX array 2x in Y and X by striding (nearest-neighbor)."""
    return array[:, :, :, ::2, ::2]


def _build_fov_roi_table(layout: FieldLayout) -> RoiTable:
    """Build a Fractal/``ngio``-style ``"FOV_ROI_table"`` recording each
    original field of view's pixel region within the stitched mosaic.

    Downstream per-FOV processing (e.g. illumination correction, which must
    be computed against the original camera frame, not an arbitrary crop of
    the stitched canvas) reads this table to recover field boundaries that
    are otherwise lost once fields are merged into one contiguous array.
    Not part of the core NGFF spec -- a Fractal/``ngio`` convention, see
    ``fractal_tasks_core/illumination_correction.py``.

    ROIs are recorded in world (physical) coordinates -- ``ngio``'s
    ``RoiTable`` only supports serializing world-coordinate ROIs -- with
    ``z`` left unconstrained, so the same table resolves correctly against
    both the full z-stack and the MIP image of the same well.

    Parameters
    ----------
    layout
        As returned by :func:`get_field_layout`.

    Returns
    -------
    RoiTable
    """
    pixel_size_x = layout.pixel_size_x if layout.pixel_size_x and layout.pixel_size_x > 0 else 1.0
    pixel_size_y = layout.pixel_size_y if layout.pixel_size_y and layout.pixel_size_y > 0 else 1.0
    if pixel_size_x == 1.0 or pixel_size_y == 1.0:
        logger.warning(
            "Missing or non-positive pixel size in nd2 metadata; falling back to "
            "1.0 micrometer/pixel for the FOV ROI table. ROI extents will be wrong "
            "in physical units (still correct in pixel count) until this is fixed "
            "at the source."
        )

    h, w = layout.tile_shape[3], layout.tile_shape[4]
    rois = [
        Roi.from_values(
            slices={
                "y": slice(y0 * pixel_size_y, (y0 + h) * pixel_size_y),
                "x": slice(x0 * pixel_size_x, (x0 + w) * pixel_size_x),
            },
            name=name or f"FOV_{i}",
            label=i,
            space="world",
        )
        for i, ((y0, x0), name) in enumerate(zip(layout.offsets, layout.position_names))
    ]
    return RoiTable(rois=rois)


def ensure_plate_exists(plate_path: Union[str, Path], plate_name: str) -> OmeZarrPlate:
    """Idempotently open or create a shared OME-Zarr plate store.

    Safe to call from multiple processes: if two callers race to create the
    same plate for the first time, the loser's ``create_empty_plate`` call
    fails and falls back to opening what the winner created.

    Intended to be called once, before any per-well writers run (e.g. from
    the same serial pass that discovers .nd2 files and generates per-well
    PBS jobscripts), rather than from every worker -- ``atomic_add_image``
    (used by :func:`convert_individual_nd2_to_ome_ngff`) only guards
    concurrent *modification* of an existing plate, not its first creation.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    plate_name
        Name recorded in the plate's metadata (only used if the store does
        not already exist).

    Returns
    -------
    OmeZarrPlate
        Opened for writing (``mode="r+"``).
    """
    try:
        return open_ome_zarr_plate(store=str(plate_path), mode="r+")
    except NgioFileNotFoundError:
        pass

    try:
        return create_empty_plate(store=str(plate_path), name=plate_name, ngff_version=NGFF_VERSION, overwrite=False)
    except NgioFileExistsError:
        return open_ome_zarr_plate(store=str(plate_path), mode="r+")


def locate_well(plate_path: Union[str, Path], well_name: str) -> str:
    """Resolve a well name to its path within a plate store.

    Prints (and returns) the well's path so it can be used directly in an
    ``rsync``/``scp`` command run from the *local* machine pulling from the
    server -- this function only resolves a path, it never itself shells
    out to copy anything, since the server generally has no outbound
    SSH/credentials to reach an arbitrary client.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    well_name
        e.g. ``"C09"``.

    Returns
    -------
    str
        The well's path relative to the plate store, e.g. ``"C/09"``.
    """
    well_match = _WELL_NAME_RE.match(well_name)
    if well_match is None:
        raise ValueError(f"Could not parse well name {well_name!r}, expected e.g. 'C09'")
    row, column = well_match.group(1).upper(), int(well_match.group(2))

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    well_path = plate.meta.get_well_path(row=row, column=column)
    full_path = str(Path(plate_path) / well_path)
    print(full_path)
    return full_path


def open_well_image(plate_path: Union[str, Path], well_relative_path: str, kind: Literal["stack", "mip"]) -> BioImage:
    """Open one well's "stack" or "mip" image (e.g. ``plate.zarr/C/09/mip``)
    as a ``BioImage``.

    ``BioImage(path)``'s automatic plugin resolution is extension-based (it
    only tries the ``bioio-ome-zarr`` reader for paths ending in ``.zarr``,
    ``.ozx``, or ``.zip`` -- confirmed by reading its `ReaderMetadata`), so it
    never matches a well's path, which sits *inside* a plate store rather
    than being its own ``.zarr``-suffixed directory. Passing the reader
    explicitly bypasses that extension check. Verified against a real
    stitched well: ``.dims``, ``.channel_names``, ``.physical_pixel_sizes``,
    and ``get_image_data("YX", C=c, Z=z)`` all round-trip correctly this way.

    Parameters
    ----------
    plate_path
        Full path to the shared plate .zarr store.
    well_relative_path
        The well's path relative to the plate store, e.g. ``"C/09"`` (as
        returned by :func:`locate_well`).
    kind
        Which image to open -- "stack" (full z-stack) or "mip" (maximum
        intensity projection). Either, both, or neither may have been
        written for a given well (see ``keep_stacks``/``mip`` on
        :func:`convert_individual_nd2_to_ome_ngff`).

    Returns
    -------
    BioImage

    Raises
    ------
    FileNotFoundError
        If ``kind`` was not written for this well.
    """
    well_path = Path(plate_path) / well_relative_path
    image_path = well_path / kind
    if not image_path.exists():
        well_group = zarr.open_group(str(well_path), mode="r")
        available = [image["path"] for image in well_group.attrs["ome"]["well"]["images"]]
        other_flag = "mip=True" if kind == "mip" else "keep_stacks=True"
        raise FileNotFoundError(
            f"Well {well_relative_path!r} has no {kind!r} image (has: {available!r}). "
            f"Re-run conversion with {other_flag} to add it."
        )
    return BioImage(str(image_path), reader=OmeZarrReader)


def convert_individual_nd2_to_ome_ngff(
    in_file_path: Union[str, Path],
    plate_path: Union[str, Path],
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    num_levels: int = NUM_PYRAMID_LEVELS,
) -> None:
    """Stitch one nd2 file's fields of view into a well image in a shared
    OME-NGFF plate store.

    Parameters
    ----------
    in_file_path
        Full path to the .nd2 image file (one well).
    plate_path
        Full path to the shared plate .zarr store. Must already exist --
        see :func:`ensure_plate_exists`.
    keep_stacks
        Whether to write the full z-stack image (at image path "stack").
    mip
        Whether to write a maximum-intensity-projection image (at image
        path "mip").
    y_direction
        Direction of increasing (stage) y-coordinates ("up" or "down").
    x_direction
        Direction of increasing (stage) x-coordinates ("left" or "right").
        See :func:`get_field_layout` for why this exists alongside
        ``y_direction``.
    placement
        "grid" or "continuous" -- see :func:`get_field_layout`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect.
    num_levels
        Number of pyramid levels to write.
    """
    if not keep_stacks and not mip:
        logger.error("Neither keep_stacks nor mip are true. Nothing will be written.")
        return

    logger.info(f"Reading layout for {in_file_path}")
    layout = get_field_layout(in_file_path, y_direction=y_direction, x_direction=x_direction, placement=placement)

    if channel_names is None:
        channel_names = layout.channel_names
    elif isinstance(channel_names, str):
        channel_names = [channel_names]

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r+")

    images = BioImage(str(in_file_path))

    if keep_stacks:
        _write_well_image(
            images=images,
            layout=layout,
            plate=plate,
            plate_path=plate_path,
            image_path="stack",
            channel_names=channel_names,
            num_levels=num_levels,
            project_z=False,
        )

    if mip:
        _write_well_image(
            images=images,
            layout=layout,
            plate=plate,
            plate_path=plate_path,
            image_path="mip",
            channel_names=channel_names,
            num_levels=num_levels,
            project_z=True,
        )


def _write_well_image(
    images: BioImage,
    layout: FieldLayout,
    plate: OmeZarrPlate,
    plate_path: Union[str, Path],
    image_path: str,
    channel_names: List[str],
    num_levels: int,
    project_z: bool,
) -> None:
    """Register and write one image (full stack or MIP) for a well, then
    attach a "FOV_ROI_table" recording each original field's pixel region
    within the stitched canvas (see :func:`_build_fov_roi_table`)."""
    canvas_shape = layout.canvas_shape
    if project_z:
        canvas_shape = (canvas_shape[0], canvas_shape[1], 1, canvas_shape[3], canvas_shape[4])

    logger.info(f"Registering well {layout.row}{layout.column:02d} image '{image_path}' in {plate_path}")
    well_relative_path = plate.atomic_add_image(row=layout.row, column=layout.column, image_path=image_path)

    image_name = f"{layout.row}{layout.column:02d}" + ("_mip" if project_z else "")
    attributes = _build_ngff_v05_metadata(
        image_name=image_name,
        num_levels=num_levels,
        pixel_size_x=layout.pixel_size_x,
        pixel_size_y=layout.pixel_size_y,
        pixel_size_z=layout.pixel_size_z,
        channel_names=channel_names,
        channel_colors=layout.channel_colors,
    )
    attributes["blimp"] = {"image_kind": "mip" if project_z else "stack"}

    root_store = zarr.storage.LocalStore(str(plate_path))
    image_group = zarr.open_group(
        store=root_store,
        path=well_relative_path,
        mode="a",
        zarr_format=3,
        attributes=attributes,
    )

    level_shapes = []
    for level in range(num_levels):
        factor = 2**level
        # Ceiling division, matching what _downsample_yx's striding actually
        # produces (e.g. a 4691-pixel axis strided by 2 yields 2346 pixels,
        # not floor(4691/2) = 2345). zarr's array assignment silently clips a
        # too-large source to the destination shape rather than raising, so a
        # floor-divided level_shape here would quietly drop the last
        # row/column at every level instead of failing loudly.
        level_shapes.append(
            (
                canvas_shape[0],
                canvas_shape[1],
                canvas_shape[2],
                max(1, -(-canvas_shape[3] // factor)),
                max(1, -(-canvas_shape[4] // factor)),
            )
        )

    level_arrays = []
    for level, level_shape in enumerate(level_shapes):
        chunk_shape = (
            1,
            1,
            1,
            min(layout.tile_shape[3], level_shape[3]),
            min(layout.tile_shape[4], level_shape[4]),
        )
        level_arrays.append(
            image_group.create_array(
                name=str(level),
                shape=level_shape,
                dtype=images.dtype,
                chunks=chunk_shape,
                shards=None,
                dimension_names=["t", "c", "z", "y", "x"],
            )
        )

    for field_index, (y0, x0) in enumerate(layout.offsets):
        logger.debug(f"Writing field {field_index} at offset ({y0}, {x0})")
        images.set_scene(field_index)
        tile = images.get_image_data("TCZYX")
        if project_z:
            tile = np.max(tile, axis=2, keepdims=True)

        h, w = tile.shape[3], tile.shape[4]
        level_arrays[0][:, :, :, y0 : y0 + h, x0 : x0 + w] = tile

    logger.debug("Building pyramid levels")
    current = level_arrays[0][:, :, :, :, :]
    for level in range(1, num_levels):
        current = _downsample_yx(current)
        level_arrays[level][:, :, :, :, :] = current

    logger.debug(f"Attaching FOV_ROI_table to '{image_path}'")
    container = open_ome_zarr_container(str(Path(plate_path) / well_relative_path))
    container.add_table("FOV_ROI_table", _build_fov_roi_table(layout), overwrite=True)


def nd2_to_ome_ngff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    plate_name: Optional[str] = None,
    n_batches: int = 1,
    batch_id: int = 0,
    keep_stacks: bool = False,
    mip: bool = False,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
) -> None:
    """Read a folder of nd2 files (one well each) and stitch them into a
    shared OME-NGFF plate store. Can perform batch processing.

    Note: this does not itself call :func:`ensure_plate_exists` for every
    file -- it assumes the plate store already exists (call
    :func:`ensure_plate_exists` once beforehand, e.g. from the same serial
    step that discovers input files and generates per-batch jobs) so that
    concurrent batches never race to create it.

    Parameters
    ----------
    in_path
        Full path to the folder of .nd2 image files.
    plate_path
        Full path to the shared plate .zarr store.
    plate_name
        Name for the plate, used only if it does not already exist (see
        :func:`ensure_plate_exists`).
    n_batches
        Number of batches into which the processing should be split.
    batch_id
        Current batch to process.
    keep_stacks
        Whether to write full z-stack images.
    mip
        Whether to write maximum-intensity-projection images.
    y_direction
        Direction of increasing (stage) y-coordinates ("up" or "down").
    x_direction
        Direction of increasing (stage) x-coordinates ("left" or "right").
    placement
        "grid" or "continuous" -- see :func:`get_field_layout`.
    channel_names
        List of channel names in case those found in the image metadata
        are incorrect and need to be replaced.
    """
    in_path = Path(in_path)
    plate_path = Path(plate_path)

    ensure_plate_exists(plate_path, plate_name or plate_path.stem)

    filename_list = _get_list_of_files_current_batch(in_path=in_path, n_batches=n_batches, batch_id=batch_id)

    logger.info(f"Converting nd2 files to OME-NGFF: {filename_list}")
    for f in filename_list:
        in_file_path = in_path / f
        convert_individual_nd2_to_ome_ngff(
            in_file_path=in_file_path,
            plate_path=plate_path,
            keep_stacks=keep_stacks,
            mip=mip,
            y_direction=y_direction,
            x_direction=x_direction,
            placement=placement,
            channel_names=channel_names,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="nd2_to_ome_ngff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert a folder of .nd2 files to OME-NGFF")
    convert_parser.add_argument("-i", "--in_path", help="directory containing the input .nd2 files", required=True)
    convert_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    convert_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    convert_parser.add_argument(
        "--batch",
        nargs=2,
        default=[1, 0],
        metavar=("N_BATCHES", "BATCH_ID"),
        help="number of batches and the current batch to process (0-indexed)",
    )
    convert_parser.add_argument(
        "--keep_stacks",
        default=False,
        action="store_true",
        help="whether to write full z-stack images (default = False)",
    )
    convert_parser.add_argument(
        "-m", "--mip", default=False, action="store_true", help="whether to write maximum intensity projections"
    )
    convert_parser.add_argument(
        "-y",
        "--y_direction",
        default="down",
        help='direction of increasing (stage) y-coordinates ("up" or "down")',
    )
    convert_parser.add_argument(
        "-x",
        "--x_direction",
        default="left",
        help='direction of increasing (stage) x-coordinates ("left" or "right")',
    )
    convert_parser.add_argument(
        "--placement",
        default="grid",
        choices=["grid", "continuous"],
        help='"grid" snaps fields to an exact tile grid (default); "continuous" uses raw stage offsets',
    )
    convert_parser.add_argument(
        "-c", "--channel_names", type=str, nargs="+", default=None, help="list of channel names"
    )
    convert_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    ensure_parser = subparsers.add_parser("ensure-plate", help="Idempotently create a plate store if it doesn't exist")
    ensure_parser.add_argument("-o", "--plate_path", help="path to the shared plate .zarr store", required=True)
    ensure_parser.add_argument("--plate_name", default=None, help="name for the plate (default: derived from path)")
    ensure_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    locate_parser = subparsers.add_parser("locate-well", help="Print the on-disk path of one well")
    locate_parser.add_argument("-p", "--plate_path", help="path to the shared plate .zarr store", required=True)
    locate_parser.add_argument("-w", "--well", help="well name, e.g. C09", required=True)
    locate_parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity (e.g. -vvv)")

    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "convert":
        nd2_to_ome_ngff(
            in_path=args.in_path,
            plate_path=args.plate_path,
            plate_name=args.plate_name,
            n_batches=args.batch[0],
            batch_id=args.batch[1],
            keep_stacks=args.keep_stacks,
            mip=args.mip,
            y_direction=args.y_direction,
            x_direction=args.x_direction,
            placement=args.placement,
            channel_names=args.channel_names,
        )
    elif args.command == "ensure-plate":
        ensure_plate_exists(args.plate_path, args.plate_name or Path(args.plate_path).stem)
    elif args.command == "locate-well":
        locate_well(args.plate_path, args.well)
