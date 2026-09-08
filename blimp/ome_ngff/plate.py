"""Shared OME-Zarr plate/well registration and image writing, used by every
OME-NGFF writer (nd2-sourced, TIFF-sourced, and future Operetta-sourced)."""
from typing import Any, Dict, List, Union, Literal, Callable, Optional
from pathlib import Path
from dataclasses import field, dataclass
import io
import os
import html
import time
import socket
import string
import logging
import functools
import itertools
import threading
import http.server
import urllib.parse

from ngio import (
    OmeZarrContainer,
    create_empty_plate,
    NgioFileExistsError,
    open_ome_zarr_plate,
    NgioFileNotFoundError,
    open_ome_zarr_container,
)
from bioio import BioImage
from bioio_ome_zarr import Reader as OmeZarrReader
from ngio.hcs._plate import OmeZarrPlate
import zarr
import numpy as np
import pandas as pd
import dask.array as da

from blimp.ome_ngff.labels import well_label_offset
from blimp.ome_ngff.layout import FieldLayout, _WELL_NAME_RE, _build_fov_roi_table
from blimp.ome_ngff.metadata import (
    NGFF_VERSION,
    _downsample_yx,
    _build_ngff_v05_metadata,
)

logger = logging.getLogger(__name__)

_PLATE_ROWS = {"96": list(string.ascii_uppercase[:8]), "384": list(string.ascii_uppercase[:16])}
_PLATE_COLUMNS = {"96": list(range(1, 13)), "384": list(range(1, 25))}


def resolve_plate_path(plate_path: Union[str, Path]) -> Path:
    """Resolve a user-given path to the actual plate .zarr store location.

    A path already ending in ``.zarr`` is used as-is; anything else is
    treated as a parent directory, with the store placed at
    ``<path>/plate.zarr``. Pass the result to :func:`ensure_plate_exists`
    (using its own ``.stem`` as the plate name) so a bare folder still gets
    a sensible derived name rather than the literal string "plate".

    Parameters
    ----------
    plate_path
        User-given path, e.g. from a CLI's ``-o``/``--plate_path``.

    Returns
    -------
    Path
        ``plate_path`` itself if it ends in ``.zarr``, else
        ``plate_path/plate.zarr``.
    """
    plate_path = Path(plate_path)
    return plate_path if plate_path.suffix == ".zarr" else plate_path / "plate.zarr"


def ensure_plate_exists(
    plate_path: Union[str, Path], plate_name: str, plate_size: Literal["96", "384"] = "384"
) -> OmeZarrPlate:
    """Idempotently open or create a shared OME-Zarr plate store.

    Safe to call from multiple processes: if two callers race to create the
    same plate, the loser's ``create_empty_plate`` call fails and falls
    back to opening what the winner created. Pre-declares the full
    row/column grid up front (costs nothing in storage until a well is
    actually written), so a viewer can place populated wells at their true
    grid position rather than only the wells a given run happened to
    touch. Call once, before any per-well writers run --
    ``atomic_add_image`` (used by the per-source writers) only guards
    concurrent *modification* of an existing plate, not its first creation.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    plate_name
        Name recorded in the plate's metadata (only used if the store does
        not already exist).
    plate_size
        "96" or "384" -- which standard plate row/column grid to declare
        (only used if the store does not already exist).

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
        plate = create_empty_plate(store=str(plate_path), name=plate_name, ngff_version=NGFF_VERSION, overwrite=False)
    except NgioFileExistsError:
        try:
            return open_ome_zarr_plate(store=str(plate_path), mode="r+")
        except NgioFileNotFoundError as e:
            # create_empty_plate refused because the path is non-empty, but
            # what's there isn't a valid OME-Zarr plate store either (e.g.
            # plate_path is an unrelated, already-populated directory) --
            # surface one clear error instead of this confusing pairing.
            raise FileExistsError(
                f"{plate_path} already exists and is not empty, but does not contain a valid "
                "OME-Zarr plate store. Pass a path to a new location, or to an existing plate "
                "store created by this same pipeline. If you're sure it's safe to remove and "
                f"want to rebuild fresh at this location, delete it yourself first: rm -rf {plate_path}"
            ) from e

    for row in _PLATE_ROWS[plate_size]:
        plate.add_row(row)
    for column in _PLATE_COLUMNS[plate_size]:
        plate.add_column(column)
    return plate


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


class _NoTrailingSlashHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """A static-file handler whose directory listings link to child
    directories by their bare name, not ``name + "/"``, and that stays
    quiet on every ordinary request.

    Every standard directory listing (this stdlib handler included, and
    Apache/nginx autoindex the same way) links to a subdirectory with a
    trailing slash -- ordinary, correct HTTP practice. But
    ``fsspec.implementations.http.HTTPFileSystem.ls()`` passes those link
    names straight through into the keys zarr's own ``FsspecStore.list_dir``
    returns, and ngio's AnnData-backed table reader
    (``ngio.tables.backends._anndata_utils.custom_anndata_read_zarr``) does
    an exact-string membership test against bare element names ("X", "obs",
    "var", ...) -- so every table sub-element is silently filtered out,
    and a ``FeatureTable`` reads back empty (``anndata``'s ``X is None``)
    over a plain directory-listing server, the identical store reading
    correctly from a local path. Only the trailing slash actually needs to
    go for this reader's benefit; nothing else here parses these listings
    by hand.

    ``BaseHTTPRequestHandler``'s own default also logs every single request
    to stderr -- fine for a handful of files, but a real plate touches many
    chunks and metadata probes per view, easily flooding an on-demand
    session's notebook output at well over 100 lines/second. Silenced
    unconditionally: this is meant to run unattended for the length of a
    viewing session, not to be watched.

    ``BaseHTTPRequestHandler``'s own default ``protocol_version`` is
    ``"HTTP/1.0"``, which closes the connection after every single request
    -- forcing a brand new TCP (and, over an SSH tunnel, a brand new
    forwarded-channel handshake) for each and every chunk/metadata request,
    rather than reusing one already-open connection. ``"HTTP/1.1"`` enables
    persistent connections instead, which fsspec's own ``HTTPFileSystem``
    (backed by a connection-pooling ``aiohttp`` session) is already willing
    to reuse -- it was only this server's own default forcing a fresh
    connection every time. See ``summarize_request_log`` for how to confirm
    this is actually the dominant cost for a given plate before assuming it.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def list_directory(self, path: Union[str, "os.PathLike[str]"]):
        try:
            entries = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None
        entries.sort(key=lambda name: name.lower())

        rows = ["<!DOCTYPE HTML><html><body><ul>"]
        for name in entries:
            rows.append(f'<li><a href="{urllib.parse.quote(name)}">{html.escape(name)}</a></li>')
        rows.append("</ul></body></html>")

        encoded = "\n".join(rows).encode("utf-8", "surrogateescape")
        f = io.BytesIO(encoded)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f


@dataclass
class ServedPlate:
    """Handle for a background HTTP server started by :func:`serve_plate_over_http`.

    On construction, prints the exact ssh command(s) to run to reach this
    server from a laptop -- see :func:`serve_plate_over_http`'s own
    docstring for why this is worth printing rather than left for the
    caller to work out by hand.
    """

    url: str
    host: str
    port: int
    login_alias: str
    _httpd: http.server.ThreadingHTTPServer = field(repr=False, compare=False)
    _thread: threading.Thread = field(repr=False, compare=False)
    request_log: List[Dict[str, Any]] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        compute_node = socket.gethostname()
        print(
            f"Serving {self.url!r} (this session's own compute node: {compute_node!r}).\n"
            "\n"
            "From your laptop, try this first (direct multi-hop tunnel):\n"
            f"    ssh -J {self.login_alias} -L {self.port}:localhost:{self.port} {compute_node}\n"
            "\n"
            "If that's refused (direct SSH to the compute node may be blocked), instead run\n"
            "this from an interactive terminal in this session:\n"
            f"    ssh -N -R {self.port}:localhost:{self.port} {self.login_alias}\n"
            "and this from your laptop:\n"
            f"    ssh -L {self.port}:localhost:{self.port} {self.login_alias}\n"
            "\n"
            "Then, in view_remote_plate_in_napari.ipynb:\n"
            f'    PLATE_PATH = "http://localhost:{self.port}/{self.url.rsplit("/", 1)[-1]}"\n'
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Shut down the background server and free its port."""
        self._httpd.shutdown()
        self._thread.join(timeout=timeout)
        self._httpd.server_close()

    def __enter__(self) -> "ServedPlate":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


def serve_plate_over_http(
    plate_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = 0,
    login_alias: str = "kdm",
    log_requests: bool = False,
) -> ServedPlate:
    """Serve ``plate_path``'s parent directory as static files over plain
    HTTP, in a daemon thread tied to this process's own lifetime, so a
    remote napari (reached via an SSH tunnel terminating on this same
    host:port) can open it live via ``store=served.url`` -- without copying
    the store down first, and without this process needing any outbound
    SSH/credentials to reach the client (mirrors :func:`locate_well`'s own
    reasoning, for the "stream live" case instead of "copy a well down").

    Every chunk blimp writes is its own file (writers here never enable
    zarr sharding -- see ``_write_well_image``'s ``shards=None``), so a
    plain ``ThreadingHTTPServer`` doing one whole-file GET per requested
    chunk is sufficient; no HTTP Range-request support is implemented or
    needed. ``ngio.open_ome_zarr_plate``/``open_ome_zarr_container`` (and
    therefore every viewing helper in this module and in
    ``blimp.napari_utils``) already accept a plain ``http://...`` string as
    ``store=`` with no further changes -- it's resolved via ``fsspec``
    automatically by zarr's own store-construction machinery.

    Uses :class:`_NoTrailingSlashHTTPRequestHandler`, not the stdlib's own
    ``SimpleHTTPRequestHandler``, so that ``FeatureTable``/``GenericRoiTable``
    reads (AnnData-backed, and so read via a directory *listing* rather than
    known sub-paths, unlike a plain image/label array) work over the served
    store too -- see that class's own docstring for why an ordinary
    directory listing silently breaks them otherwise.

    On success, the returned :class:`ServedPlate` prints the exact
    command(s) to run next -- both a direct multi-hop tunnel (tried first)
    and a reverse-tunnel-via-login-node fallback -- pre-filled with this
    session's own hostname and the bound port, so there's nothing to
    manually substitute.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    host
        Bind address. Defaults to ``127.0.0.1`` -- deliberately never
        ``0.0.0.0``. This is meant to be reached only through an
        authenticated SSH tunnel terminating on this same machine, never
        exposed directly on the cluster's internal network.
    port
        TCP port to bind. ``0`` (the default) asks the OS for a free port
        -- read it back from the returned ``ServedPlate.port``/``.url``.
        Pass an explicit port instead if you'd rather hardcode one number
        into both this call and your own tunnel commands.
    login_alias
        The cluster's login node, as configured in the *user's own* local
        ssh config (default ``"kdm"``) -- change it if you use a different
        alias, or pass your own full ``user@host`` if you have none
        configured.
    log_requests
        Record every request's path, server-side handling time, and which
        underlying connection it rode on, into the returned
        ``ServedPlate.request_log`` -- pass to :func:`summarize_request_log`
        for a breakdown by node kind (table/label/image) and request kind
        (metadata/chunk/listing), and to check whether persistent
        connections are actually being reused. Off by default: a normal
        viewing session has no need for it, only a one-off performance
        investigation does.

    Returns
    -------
    ServedPlate
        ``.url`` is the exact string to pass as ``store=`` from the
        reading side of the tunnel. Call ``.stop()`` (or use as a context
        manager) to shut it down early; otherwise it runs, as a daemon
        thread, until this kernel process exits -- nothing to separately
        clean up if the on-demand session is killed or times out.
    """
    plate_path = Path(plate_path)
    request_log: List[Dict[str, Any]] = []
    connection_ids = itertools.count()

    class _Handler(_NoTrailingSlashHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._connection_id = next(connection_ids)
            super().__init__(*args, **kwargs)

        def do_GET(self) -> None:
            if not log_requests:
                super().do_GET()
                return
            start = time.monotonic()
            super().do_GET()
            request_log.append(
                {
                    "path": self.path,
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "connection_id": self._connection_id,
                }
            )

    handler = functools.partial(_Handler, directory=str(plate_path.parent))
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    bound_port = httpd.server_address[1]
    return ServedPlate(
        url=f"http://{host}:{bound_port}/{plate_path.name}",
        host=host,
        port=bound_port,
        login_alias=login_alias,
        _httpd=httpd,
        _thread=thread,
        request_log=request_log,
    )


def _categorize_request_path(path: str) -> str:
    """Which kind of on-disk node a served request path belongs to.

    "table" (an AnnData-backed ``FeatureTable``/``GenericRoiTable``, stored
    as many small per-column sub-arrays), "label" (a segmentation label
    array), or "image" (the intensity image, or a plate/well/root group) --
    matching blimp's own on-disk layout (``.../tables/<name>/...``,
    ``.../labels/<name>/...``)."""
    if "/tables/" in path:
        return "table"
    if "/labels/" in path:
        return "label"
    return "image"


def summarize_request_log(served: ServedPlate) -> pd.DataFrame:
    """Break down a :class:`ServedPlate`'s ``request_log`` (only populated
    if it was started with ``serve_plate_over_http(..., log_requests=True)``)
    by node kind (table/label/image) and request kind (metadata file vs
    chunk data vs directory listing) -- meant to answer *what* a slow remote
    view actually spent its requests on before picking a fix (e.g. dropping
    the finest pyramid level or converting to 8-bit only helps "image"/
    "label" chunk cost; not loading labels initially only helps "label";
    neither helps "table", which is dominated by request *count* -- an
    AnnData-backed table stores every column as its own small array, so one
    well's feature table alone can be dozens of small requests).

    Also prints the requests-per-connection ratio -- the tell for whether
    persistent connections (``_NoTrailingSlashHTTPRequestHandler``'s
    ``protocol_version = "HTTP/1.1"``) are actually being reused: close to 1
    means every single request paid a fresh connection handshake (expensive
    over a real SSH tunnel, cheap on a local loopback test); well above 1
    means connections are being reused and the cost lies elsewhere.

    Raises
    ------
    ValueError
        If ``served.request_log`` is empty (``log_requests`` was not
        ``True``, or nothing has requested anything yet).
    """
    if not served.request_log:
        raise ValueError("request_log is empty -- pass log_requests=True to serve_plate_over_http to collect it.")
    df = pd.DataFrame(served.request_log)
    df["node_kind"] = df["path"].map(_categorize_request_path)
    df["request_kind"] = np.where(
        df["path"].str.endswith(("zarr.json", ".zattrs", ".zgroup", ".zmetadata")),
        "metadata",
        np.where(df["path"].str.contains("/c/"), "chunk", "listing"),
    )

    total_requests = len(df)
    total_connections = df["connection_id"].nunique()
    print(
        f"{total_requests} requests over {total_connections} connections "
        f"({total_requests / total_connections:.1f} requests/connection -- "
        "close to 1 means persistent connections are not actually being reused)."
    )
    return df.groupby(["node_kind", "request_kind"]).agg(requests=("path", "count"), total_ms=("duration_ms", "sum"))


def open_well_image(plate_path: Union[str, Path], well_relative_path: str, kind: Literal["stack", "mip"]) -> BioImage:
    """Open one well's "stack" or "mip" image (e.g. ``plate.zarr/C/09/mip``)
    as a ``BioImage``.

    ``BioImage(path)``'s automatic plugin resolution is extension-based (it
    only tries the ``bioio-ome-zarr`` reader for paths ending in ``.zarr``,
    ``.ozx``, or ``.zip``), so it never matches a well's path, which sits
    *inside* a plate store rather than being its own ``.zarr``-suffixed
    directory. Passing the reader explicitly bypasses that extension check.

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
        written for a given well.

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


def _discover_wells_with_label(
    plate: OmeZarrPlate,
    kind: Literal["stack", "mip"],
    label_name: Optional[str] = None,
    wells: Optional[Union[str, List[str]]] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
) -> Dict[str, OmeZarrContainer]:
    """Open every well's own ``kind`` image, restricted to wells that have
    ``label_name`` (if given) and, optionally, to an explicit ``wells``
    subset -- shared by ``build_plate_pyramid``, ``build_feature_pyramid``,
    and ``_read_plate_wide_features``.

    Parameters
    ----------
    plate
        The already-open plate.
    kind
        "stack" or "mip". Wells that don't have this image are skipped.
    label_name
        Only keep wells that have this label. ``None`` keeps every well
        that has the ``kind`` image, regardless of labels.
    wells
        Restrict discovery to one well path (e.g. ``"C/09"``, the same
        format ``plate.wells_paths()`` returns) or a list of them. ``None``
        (the default) considers every well the plate has.
    open_containers
        Already-open containers to filter/reuse instead of calling
        ``plate.get_image()`` again per well. Passed by ``add_plate``,
        which already opens every well's container itself -- reusing them
        here avoids each well paying a real, separate re-open (roughly a
        ~4x-per-well redundant-request cost over a remote store) once for
        its own image pyramid, again for every label's pyramid, and again
        for every label's feature table. ``None`` (the default) opens each
        well itself, same as every caller other than ``add_plate``.

    Returns
    -------
    Dict[str, OmeZarrContainer]
        Keyed by well path (e.g. ``"C/09"``).

    Raises
    ------
    ValueError
        If ``wells`` names a well path not in ``plate.wells_paths()``.
    """
    all_well_paths = plate.wells_paths()
    if wells is None:
        well_paths = all_well_paths
    else:
        requested = [wells] if isinstance(wells, str) else list(wells)
        unknown = sorted(set(requested) - set(all_well_paths))
        if unknown:
            raise ValueError(f"Well(s) {unknown} not found in plate; available wells: {sorted(all_well_paths)}")
        well_paths = requested

    containers: Dict[str, OmeZarrContainer] = {}
    for well_path in well_paths:
        if open_containers is not None:
            container = open_containers.get(well_path)
            if container is None:
                continue
        else:
            row, column = well_path.split("/")
            try:
                container = plate.get_image(row, column, kind)
            except ValueError:
                continue
        if label_name is not None and label_name not in container.list_labels():
            continue
        containers[well_path] = container
    return containers


def build_plate_pyramid(
    plate_path: Union[str, Path],
    kind: Literal["stack", "mip"] = "mip",
    label_name: Optional[str] = None,
    gap_fraction: float = 0.05,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
) -> List[da.Array]:
    """Every pyramid level of the whole plate, as one lazy dask array per
    level, for viewing (or otherwise computing over) all wells at once at
    their true row/column position.

    Real wells are placed at their true grid position, with a small empty
    margin (``gap_fraction`` of tile size) between them; every other
    declared grid position is a zero-cost ``da.zeros`` placeholder, so cost
    scales with ``O(populated wells)``, not the declared grid size --
    unlike ``ome_zarr.reader``'s own whole-plate stitching, which eagerly
    allocates a real zero array for every declared-but-empty position, at
    every level and channel. Every well is read through ``ngio``'s own
    ``OmeZarrPlate``/``OmeZarrContainer``/``Image`` API.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    kind
        Which image to build the pyramid from -- "stack" or "mip". Wells
        that don't have this image are skipped.
    label_name
        Build a label's pyramid instead of the intensity image's. Every
        well's own IDs get a well-specific offset first
        (:func:`blimp.ome_ngff.labels.well_label_offset`), so IDs are
        unique across the whole plate, not just within one well -- the
        canvas dtype becomes ``int64`` for this reason (wells without this
        label are skipped).
    gap_fraction
        Size of the empty margin between adjacent wells, as a fraction of
        each pyramid level's own tile size.
    plate, open_containers
        An already-open plate and/or its already-open well containers, to
        avoid reopening them from scratch -- see
        :func:`_discover_wells_with_label`'s own ``open_containers`` for
        why. ``None`` (the default) opens ``plate_path`` itself, same as
        before.

    Returns
    -------
    List[dask.array.Array]
        One array per pyramid level, finest first.

    Raises
    ------
    ValueError
        If no well has the requested image/label.
    """
    if plate is None:
        plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, open_containers=open_containers)

    if not containers:
        what = f"label {label_name!r}" if label_name is not None else f"{kind!r} image"
        raise ValueError(f"No well in {plate_path} has a {what}.")

    def _get_array(container: OmeZarrContainer, level_path: str, offset: int = 0) -> da.Array:
        # Drop the leading T axis (always size 1 for blimp's images -- these
        # are static plates, not time series); keep C/Z (or just Z, for a
        # label) exactly as ngio returns them, so this works unchanged for
        # both a MIP (Z size 1) and a full stack (Z size N).
        if label_name is not None:
            arr = container.get_label(label_name, path=level_path).get_as_dask()[0]
            if offset:
                arr = arr.astype(np.int64)
                arr = da.where(arr == 0, 0, arr + offset)
            return arr
        return container.get_image(path=level_path).get_as_dask()[0]

    n_rows, n_cols = len(plate.rows), len(plate.columns)
    row_col_index = {}
    well_offsets = {}
    for well_path in containers:
        row, column = well_path.split("/")
        row_idx, col_idx = plate.rows.index(row), plate.columns.index(column)
        row_col_index[well_path] = (row_idx, col_idx)
        well_offsets[well_path] = well_label_offset(row_idx, col_idx) if label_name is not None else 0

    first_container = next(iter(containers.values()))
    pyramid = []
    for level_path in first_container.level_paths:
        reference = _get_array(first_container, level_path)
        *leading_shape, tile_h, tile_w = reference.shape
        pitch_h = round(tile_h * (1 + gap_fraction))
        pitch_w = round(tile_w * (1 + gap_fraction))

        canvas_dtype = np.int64 if label_name is not None else reference.dtype
        canvas = da.zeros(
            (*leading_shape, n_rows * pitch_h, n_cols * pitch_w),
            dtype=canvas_dtype,
            chunks=(*leading_shape, pitch_h, pitch_w),
        )
        for well_path, container in containers.items():
            row_idx, col_idx = row_col_index[well_path]
            y0, x0 = row_idx * pitch_h, col_idx * pitch_w
            canvas[..., y0 : y0 + tile_h, x0 : x0 + tile_w] = _get_array(container, level_path, well_offsets[well_path])
        pyramid.append(canvas)

    return pyramid


def _read_plate_wide_features(
    plate_path: Union[str, Path],
    label_name: str,
    kind: Literal["stack", "mip"] = "mip",
    wells: Optional[Union[str, List[str]]] = None,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
    feature_names: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Every contributing well's own ``f"{label_name}_features"`` table,
    merged into one plate-wide dataframe with its ``"label"`` column
    rewritten to match that well's own plate-wide-unique pixel values
    (see :func:`blimp.ome_ngff.labels.well_label_offset`) -- prefers each
    well's already-persisted ``global_id_numeric`` column when present,
    else derives the identical value on the fly.

    This is the plate-wide feature-loading helper for classifier training
    (load a set of features, or every feature, across every contributing
    well) -- it's also how a caller discovers which feature names are even
    available in the first place (call once with ``feature_names=None``
    and inspect ``.columns``, as both viewing notebooks already document);
    ``ngio`` has no cheaper way to list column names, so there's no faster
    path for that discovery step. For the narrower "just one named feature,
    plus object ids, for a colormap" case, see
    :func:`_read_plate_wide_feature_raw` instead, which reads far less per
    well.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to read (``f"{label_name}_features"``).
    kind
        "stack" or "mip".
    wells
        Restrict to one well path (e.g. ``"C/09"``) or a list of them (see
        :func:`_discover_wells_with_label`). ``None`` (the default) merges
        every well that has both the label and a matching features table.
    plate, open_containers
        An already-open plate and/or its already-open well containers, to
        avoid reopening them from scratch -- see
        :func:`_discover_wells_with_label`'s own ``open_containers`` for
        why. ``None`` (the default) opens ``plate_path`` itself, same as
        before.
    feature_names
        Restrict the returned dataframe to just these columns (plus
        ``"label"``, always kept) -- useful for a training pipeline that
        wants a specific subset without carrying every column downstream.
        Applied after every well's own read, so it does not reduce the
        underlying read cost (``ngio`` has no partial-column read API --
        see :func:`_read_plate_wide_feature_raw`'s own docstring), only the
        shape of what's returned. ``None`` (the default) returns every
        column, same as before.

    Returns
    -------
    Optional[pandas.DataFrame]
        ``None`` if no (selected) well has a matching features table.
    """
    if plate is None:
        plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, wells, open_containers=open_containers)

    feature_frames = []
    for well_path, container in containers.items():
        table_name = f"{label_name}_features"
        if table_name not in container.list_tables():
            continue
        row, column = well_path.split("/")
        df = container.get_feature_table(table_name).dataframe.reset_index()
        if "global_id_numeric" in df.columns:
            df["label"] = df["global_id_numeric"]
        else:
            df["label"] = df["label"] + well_label_offset(plate.rows.index(row), plate.columns.index(column))
        if feature_names is not None:
            keep = ["label"] + [c for c in feature_names if c in df.columns and c != "label"]
            df = df[keep]
        feature_frames.append(df)

    if not feature_frames:
        return None
    return pd.concat(feature_frames, ignore_index=True)


def _read_one_feature_column_raw(table_group_path: str, feature_name: str) -> Optional[pd.DataFrame]:
    """Read one named column of an AnnData-backed FeatureTable directly via
    zarr, bypassing ``ngio.FeatureTable.dataframe``'s all-or-nothing read.

    ``ngio`` has no partial-column read API -- its own code says selecting
    columns "is not straightforward to do so for an arbitrary AnnData
    object" (``ngio/tables/backends/_abstract_backend.py``). But the
    on-disk layout is a documented, versioned external convention
    (AnnData-on-zarr's own ``encoding-type``/``encoding-version`` attrs),
    not an undocumented ``ngio`` implementation detail -- reading it
    directly here needs no private ``ngio`` attribute.

    Only reads ``var/_index`` (column names), ``obs``'s own cheap
    ``zarr.json`` attrs (its column names and which one is the index, with
    no directory listing needed), whichever of ``"label"``/
    ``"global_id_numeric"`` those name, and ``X``'s one relevant column
    slice -- skipping the directory listing and every other ``obs`` column
    ``ngio.FeatureTable.dataframe`` would otherwise fetch (roughly a
    ~172-request-per-well cost, down to roughly 10-15).

    Parameters
    ----------
    table_group_path
        Full path to the table's own zarr group, e.g.
        ``f"{plate_path}/{well_path}/{kind}/tables/{label_name}_features"``.
    feature_name
        The single measurement column to read.

    Returns
    -------
    Optional[pandas.DataFrame]
        Two or three columns (``"label"`` -- local, not yet plate-wide
        offset -- ``feature_name``, and ``"global_id_numeric"`` if
        present), or ``None`` if the table doesn't have ``feature_name`` as
        a column, or its layout doesn't match the expected v1
        ``feature_table`` shape closely enough to trust (caller should
        fall back to a full ``ngio`` read instead).
    """
    try:
        group = zarr.open_group(store=table_group_path, mode="r")
        attrs = dict(group.attrs)
        if attrs.get("type") != "feature_table" or attrs.get("table_version") != "1":
            return None

        var_index = [str(name) for name in group["var"]["_index"][:]]
        if feature_name not in var_index:
            return None
        feature_idx = var_index.index(feature_name)

        obs_attrs = dict(group["obs"].attrs)
        if obs_attrs.get("_index") != "label":
            return None
        column_order = obs_attrs.get("column-order", [])

        labels = group["obs"]["label"][:]
        if attrs.get("index_type") == "int":
            labels = labels.astype(np.int64)

        result: Dict[str, np.ndarray] = {"label": labels, feature_name: group["X"][:, feature_idx]}
        if "global_id_numeric" in column_order:
            result["global_id_numeric"] = group["obs"]["global_id_numeric"][:]
        return pd.DataFrame(result)
    except Exception:
        logger.debug(
            f"Raw feature-column read failed for {table_group_path!r}; falling back to a full read.", exc_info=True
        )
        return None


def _read_plate_wide_feature_raw(
    plate_path: Union[str, Path],
    label_name: str,
    feature_name: str,
    kind: Literal["stack", "mip"] = "mip",
    wells: Optional[Union[str, List[str]]] = None,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
) -> Optional[pd.DataFrame]:
    """Just ``label_name``'s ``feature_name`` column (plus each object's
    own plate-wide-unique ``"label"``), merged across every contributing
    well -- the narrow counterpart to :func:`_read_plate_wide_features`,
    for :func:`~blimp.napari_utils.add_feature_heatmap`'s own "one feature,
    for a colormap" case, which never needs the other measurement columns
    a full table read would otherwise fetch.

    Reads each well's table via :func:`_read_one_feature_column_raw` (plain
    zarr, skipping ``ngio.FeatureTable.dataframe``'s directory listing and
    unused ``obs`` columns), falling back to a full ``ngio``-based read for
    any individual well whose table doesn't match the expected layout
    closely enough to trust -- correctness over speed for anything
    unexpected; every other well still takes the fast path.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to read (``f"{label_name}_features"``).
    feature_name
        The single measurement column to read.
    kind
        "stack" or "mip".
    wells, plate, open_containers
        See :func:`_read_plate_wide_features` -- identical meaning.

    Returns
    -------
    Optional[pandas.DataFrame]
        Two columns, ``"label"`` (plate-wide-unique) and ``feature_name``
        -- or ``None`` if no (selected) well has a matching features table
        with this column.
    """
    if plate is None:
        plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, wells, open_containers=open_containers)
    table_name = f"{label_name}_features"

    feature_frames = []
    for well_path, container in containers.items():
        if table_name not in container.list_tables():
            continue

        table_group_path = f"{plate_path}/{well_path}/{kind}/tables/{table_name}"
        df = _read_one_feature_column_raw(table_group_path, feature_name)
        if df is None:
            full_df = container.get_feature_table(table_name).dataframe.reset_index()
            if feature_name not in full_df.columns:
                continue
            keep = ["label", feature_name] + (["global_id_numeric"] if "global_id_numeric" in full_df.columns else [])
            df = full_df[keep].copy()

        row, column = well_path.split("/")
        if "global_id_numeric" in df.columns:
            df["label"] = df["global_id_numeric"]
        else:
            df["label"] = df["label"] + well_label_offset(plate.rows.index(row), plate.columns.index(column))
        feature_frames.append(df[["label", feature_name]])

    if not feature_frames:
        return None
    return pd.concat(feature_frames, ignore_index=True)


def build_feature_pyramid(
    plate_path: Union[str, Path],
    label_name: str,
    feature_name: str,
    kind: Literal["stack", "mip"] = "mip",
    gap_fraction: float = 0.05,
    wells: Optional[Union[str, List[str]]] = None,
) -> List[da.Array]:
    """Every pyramid level of a plate-wide *feature-value* image: each
    object's own pixels hold its own ``feature_name`` measurement (a plain
    float) instead of a label ID -- independent of per-label colormap
    tools like Napari Feature Visualizer, which assume label IDs are
    packed near zero and can run out of memory on plate-wide-unique IDs
    (see :func:`blimp.ome_ngff.labels.well_label_offset`).

    Background pixels, and any object with no matching feature row, read
    as ``NaN`` (the canvas dtype is always ``float32``). Cost scales with
    the number of *populated* (and, if given, *selected*) wells, not the
    plate's declared grid size, same as ``build_plate_pyramid`` -- pass
    ``wells`` to restrict this to only the well(s) you actually need.

    Parameters
    ----------
    plate_path
        Full path to the plate's .zarr store.
    label_name
        Which label's own measurements to show (``f"{label_name}_features"``).
    feature_name
        Which column of that features table to show.
    kind
        "stack" or "mip".
    gap_fraction
        Size of the empty margin between adjacent wells, as a fraction of
        each pyramid level's own tile size (see ``build_plate_pyramid``).
    wells
        Restrict to one well path (e.g. ``"C/09"``) or a list of them (see
        :func:`_discover_wells_with_label`). ``None`` (the default)
        includes every populated well.

    Returns
    -------
    List[dask.array.Array]
        One array per pyramid level, finest first.

    Raises
    ------
    ValueError
        If no (selected) well has a ``f"{label_name}_features"`` table, or
        if ``feature_name`` isn't one of its columns.
    """
    features_df = _read_plate_wide_features(plate_path, label_name, kind, wells)
    if features_df is None:
        raise ValueError(f"No well in {plate_path} has a {label_name}_features table.")
    if feature_name not in features_df.columns:
        raise ValueError(
            f"{feature_name!r} is not a column of {label_name}_features; available: {sorted(features_df.columns)}"
        )

    order = np.argsort(features_df["label"].to_numpy())
    sorted_ids = features_df["label"].to_numpy()[order]
    sorted_values = features_df[feature_name].to_numpy(dtype=np.float32)[order]

    def _remap(block: np.ndarray) -> np.ndarray:
        flat = block.ravel()
        idx = np.clip(np.searchsorted(sorted_ids, flat), 0, len(sorted_ids) - 1)
        matched = sorted_ids[idx] == flat
        return np.where(matched, sorted_values[idx], np.nan).astype(np.float32).reshape(block.shape)

    plate = open_ome_zarr_plate(store=str(plate_path), mode="r")
    containers = _discover_wells_with_label(plate, kind, label_name, wells)

    n_rows, n_cols = len(plate.rows), len(plate.columns)
    row_col_index = {}
    well_offsets = {}
    for well_path in containers:
        row, column = well_path.split("/")
        row_idx, col_idx = plate.rows.index(row), plate.columns.index(column)
        row_col_index[well_path] = (row_idx, col_idx)
        well_offsets[well_path] = well_label_offset(row_idx, col_idx)

    first_container = next(iter(containers.values()))
    pyramid = []
    for level_path in first_container.level_paths:
        reference = first_container.get_label(label_name, path=level_path).get_as_dask()[0]
        *leading_shape, tile_h, tile_w = reference.shape
        pitch_h = round(tile_h * (1 + gap_fraction))
        pitch_w = round(tile_w * (1 + gap_fraction))

        canvas = da.full(
            (*leading_shape, n_rows * pitch_h, n_cols * pitch_w),
            np.nan,
            dtype=np.float32,
            chunks=(*leading_shape, pitch_h, pitch_w),
        )
        for well_path, container in containers.items():
            row_idx, col_idx = row_col_index[well_path]
            y0, x0 = row_idx * pitch_h, col_idx * pitch_w
            arr = container.get_label(label_name, path=level_path).get_as_dask()[0]
            arr = da.where(arr == 0, 0, arr.astype(np.int64) + well_offsets[well_path])
            canvas[..., y0 : y0 + tile_h, x0 : x0 + tile_w] = arr.map_blocks(_remap, dtype=np.float32)
        pyramid.append(canvas)

    return pyramid


def _write_well_image(
    get_tile: Callable[[int], np.ndarray],
    layout: FieldLayout,
    plate: OmeZarrPlate,
    plate_path: Union[str, Path],
    image_path: str,
    channel_names: List[str],
    channel_colors: List[str],
    dtype: Any,
    num_levels: int,
    project_z: bool,
) -> OmeZarrContainer:
    """Register and write one image (full stack or MIP) for a well, then
    attach a "FOV_ROI_table" recording each original field's pixel region
    within the stitched canvas (see :func:`blimp.ome_ngff.layout._build_fov_roi_table`).

    Source-agnostic: ``get_tile(field_index)`` supplies each field's TCZYX
    pixel array (an nd2-sourced caller wraps ``BioImage.set_scene``; a
    TIFF-sourced caller opens each field's own TIFF, or returns a blank
    array for a field whose TIFF is missing).

    Parameters
    ----------
    get_tile
        Given a 0-indexed field index (in the same order as
        ``layout.offsets``), returns that field's TCZYX pixel array.
    layout
        As returned by ``get_field_layout``/``get_field_layout_from_tiff_metadata``.
    plate
        The already-open plate (see :func:`ensure_plate_exists`).
    plate_path
        Full path to the plate's .zarr store.
    image_path
        "stack" or "mip" (or any other short, alphanumeric path -- only
        ``ngio``'s ``path_in_well_validation`` constrains this).
    channel_names, channel_colors
        One entry per channel.
    dtype
        Pixel dtype for the written arrays.
    num_levels
        Number of pyramid levels to write.
    project_z
        Whether to write a maximum-intensity projection (True) or the full
        z-stack (False).

    Returns
    -------
    OmeZarrContainer
        The newly-written image, opened -- callers reuse it to attach
        labels/features without a redundant re-open.
    """
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
        channel_colors=channel_colors,
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
                dtype=dtype,
                chunks=chunk_shape,
                shards=None,
                dimension_names=["t", "c", "z", "y", "x"],
            )
        )

    for field_index, (y0, x0) in enumerate(layout.offsets):
        logger.debug(f"Writing field {field_index} at offset ({y0}, {x0})")
        tile = get_tile(field_index)
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
    return container
