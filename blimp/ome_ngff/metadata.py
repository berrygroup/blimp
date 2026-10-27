"""NGFF 0.5 metadata construction and pyramid downsampling, shared across
every OME-NGFF writer."""
from typing import Any, Dict, List

import numpy as np

NGFF_VERSION = "0.5"
NUM_PYRAMID_LEVELS = 5


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
