"""Shared OME-Zarr plate/well registration and image writing, used by every
OME-NGFF writer (nd2-sourced, TIFF-sourced, and future Operetta-sourced)."""
from typing import Any, Dict, List, Union, Literal, TypeVar, Callable, Optional
from pathlib import Path
from dataclasses import field, dataclass
import io
import os
import html
import time
import random
import socket
import string
import logging
import functools
import itertools
import threading
import subprocess
import http.server
import urllib.parse
import concurrent.futures

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
import filelock
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

_T = TypeVar("_T")
_R = TypeVar("_R")


def _map_concurrently(func: Callable[[_T], _R], items: List[_T], max_workers: int) -> List[_R]:
    """Apply ``func`` to each of ``items``, across up to ``max_workers``
    threads, returning results in ``items`` order regardless of completion
    order.

    Every per-well read this module makes is dominated by network
    round-trip latency, not server CPU or bandwidth (confirmed against
    real cluster request logs) -- ``zarr``/``ngio``'s underlying
    ``fsspec`` HTTP store marshals each blocking call onto its own async
    event loop, so calling it from several threads at once genuinely
    overlaps their round trips instead of serializing on Python's GIL.
    ``max_workers=1`` runs sequentially in the calling thread, in
    ``items`` order, with no thread pool involved.
    """
    if max_workers <= 1:
        return [func(item) for item in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        return [future.result() for future in futures]


def _normalize_mirror_urls(plate_path: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
    """A ``plate_path`` accepted as either one path/URL or several mirror
    URLs for the same store (see :func:`open_mirrored_tunnels`), normalized
    to a plain list of strings -- a single value becomes a one-element
    list, so every mirror-aware function below behaves exactly as before
    when only one is given."""
    if isinstance(plate_path, (str, Path)):
        return [str(plate_path)]
    return [str(mirror) for mirror in plate_path]


def _well_mirror_map(all_well_paths: List[str], mirror_urls: List[str]) -> Dict[str, str]:
    """Which of ``mirror_urls`` each well path opens through, round-robined
    by its position in ``all_well_paths`` -- computed the same way
    everywhere it's needed (:func:`_discover_wells_with_label`'s own
    opening loop, ``add_plate``'s, and ``_read_plate_wide_feature_raw``'s
    raw-path construction) so a given well always resolves to the same
    mirror, regardless of which of those call it.

    Always keyed off the plate's *full* well list, not a caller's
    ``wells=`` filter -- otherwise the same well could map to a different
    mirror depending on which subset of wells a particular call happened
    to restrict itself to.
    """
    return {well_path: mirror_urls[i % len(mirror_urls)] for i, well_path in enumerate(all_well_paths)}


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


@dataclass
class MirroredTunnels:
    """Handle for several independent ``ssh -L`` tunnel processes, all
    forwarding to the same remote host:port, started by
    :func:`open_mirrored_tunnels`.

    A single ``ssh -L`` port forward multiplexes over one encrypted TCP
    connection -- fine for interactive use, but a real ceiling on
    throughput once many small requests need to move concurrently
    (confirmed against real cluster traffic: splitting the same total
    concurrency across several separate ssh connections roughly halved
    wall-clock time versus one connection carrying it all). Each of this
    handle's ``local_ports`` is its own separate ``ssh`` process -- a
    genuinely independent connection, not just another ``-L`` flag on one
    -- so per-well reads can be spread across them via
    :func:`_well_mirror_map`.

    A background thread polls each process every ``check_interval``
    seconds and restarts any that has died (network drop, laptop sleep, VPN
    reconnect) -- a tunnel dying mid-session degrades to fewer working
    mirrors for a few seconds, not a silent, permanent stall. A tunnel found
    dead ``max_consecutive_failures`` checks in a row, with no live interval
    in between, is assumed to be a persistent problem (bad host key, auth,
    or TCP forwarding disabled on the remote host) rather than a transient
    drop -- it's logged as an error, with whatever ``ssh`` itself printed to
    stderr, and left dead rather than retried forever; the other mirrors
    keep working independently. This also protects against hammering the
    login node with rapid repeated connection attempts, which some sites'
    own automated defenses (e.g. fail2ban) may respond to by temporarily
    blocking the connecting IP altogether.
    """

    local_ports: List[int]
    remote_host: str
    remote_port: int
    login_alias: str
    check_interval: float = 5.0
    max_consecutive_failures: int = 3
    _processes: List[subprocess.Popen] = field(default_factory=list, repr=False, compare=False)
    _fail_streaks: List[int] = field(default_factory=list, repr=False, compare=False)
    _given_up: List[bool] = field(default_factory=list, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _monitor_thread: Optional[threading.Thread] = field(default=None, repr=False, compare=False)

    def urls(self, plate_name: str) -> List[str]:
        """The mirror URLs to pass as ``plate_path`` to e.g. ``add_plate``/
        ``build_plate_pyramid`` -- one per tunnel. ``plate_name`` is
        whatever :func:`serve_plate_over_http` printed after the port
        (e.g. ``"my_plate.zarr"``)."""
        return [f"http://localhost:{port}/{plate_name}" for port in self.local_ports]

    def _spawn(self, local_port: int) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ssh",
                "-N",
                "-J",
                self.login_alias,
                "-L",
                f"{local_port}:localhost:{self.remote_port}",
                self.remote_host,
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "BatchMode=yes",
                # On-demand HPC compute nodes are typically freshly, dynamically
                # allocated each session -- a never-before-seen host key is the
                # normal case here, not a red flag, and BatchMode=yes above means
                # ssh can never interactively ask to accept one; accept-new does
                # so automatically (while still refusing a *changed* key for a
                # host seen before, unlike an outright StrictHostKeyChecking=no).
                "-o",
                "StrictHostKeyChecking=accept-new",
            ],
            stdout=subprocess.DEVNULL,
            # Captured (not discarded) so a dead tunnel's actual reason -- host key,
            # auth, forwarding disabled, wrong hostname -- can be logged instead of
            # just a bare exit code. Safe to read after the process has already
            # exited: ssh's own error output here is short, so the pipe never fills
            # up while nothing is reading it.
            stderr=subprocess.PIPE,
            text=True,
        )

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.check_interval):
            with self._lock:
                for i, proc in enumerate(self._processes):
                    if self._given_up[i]:
                        continue
                    if proc.poll() is None:
                        self._fail_streaks[i] = 0  # alive and well at this check
                        continue

                    stderr = (proc.stderr.read() or "").strip() if proc.stderr is not None else ""
                    reason = f": {stderr}" if stderr else " (no stderr output)"
                    self._fail_streaks[i] += 1

                    if self._fail_streaks[i] >= self.max_consecutive_failures:
                        self._given_up[i] = True
                        logger.error(
                            f"Tunnel on local port {self.local_ports[i]} failed "
                            f"{self._fail_streaks[i]} times in a row (code {proc.returncode}){reason} -- "
                            "giving up on this mirror rather than retrying forever; the "
                            "other mirrors are unaffected. This usually means a persistent "
                            "problem (host key, auth, or TCP forwarding disabled on the "
                            "remote host), not a transient drop -- check the error above."
                        )
                        continue

                    logger.warning(
                        f"Tunnel on local port {self.local_ports[i]} exited "
                        f"(code {proc.returncode}){reason}; restarting."
                    )
                    self._processes[i] = self._spawn(self.local_ports[i])

    def stop(self, timeout: float = 5.0) -> None:
        """Stop monitoring and terminate every tunnel process."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=timeout)
        with self._lock:
            for proc in self._processes:
                proc.terminate()
            for proc in self._processes:
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def __enter__(self) -> "MirroredTunnels":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


def open_mirrored_tunnels(
    remote_host: str,
    remote_port: int,
    login_alias: str = "kdm",
    n_mirrors: int = 4,
    local_ports: Optional[List[int]] = None,
    check_interval: float = 5.0,
    max_consecutive_failures: int = 3,
) -> MirroredTunnels:
    """Start ``n_mirrors`` independent ``ssh -L`` tunnels from this machine
    to ``remote_host:remote_port`` (the same remote port
    :func:`serve_plate_over_http` printed), each on its own local port, so
    reads can be spread across genuinely separate SSH connections instead
    of hitting one connection's own multiplexing ceiling -- see
    :class:`MirroredTunnels` for why this matters and how a dropped tunnel
    is handled. Run this on the machine that will actually view the plate
    (i.e. from ``view_remote_plate_in_napari.ipynb`` itself, not the
    cluster-side serving notebook) -- it needs its own outbound SSH access
    to the cluster, same as the single-tunnel instructions
    :func:`serve_plate_over_http` prints.

    Parameters
    ----------
    remote_host
        The compute node hostname :func:`serve_plate_over_http` printed.
    remote_port
        The port :func:`serve_plate_over_http` printed (the same number
        the single-tunnel instructions already use).
    login_alias
        Your ssh config's ``Host`` alias for the cluster's login node
        (passed to ``-J``) -- see :func:`serve_plate_over_http`.
    n_mirrors
        How many independent tunnels to open.
    local_ports
        Explicit local ports to use, one per mirror -- ``None`` (the
        default) picks ``n_mirrors`` free ports automatically (briefly
        binding each to find one, then releasing it for ``ssh`` to bind in
        turn -- a small, generally-safe race, same tradeoff ``port=0``
        makes elsewhere in this module).
    check_interval
        How often (seconds) the background monitor thread checks for a
        dead tunnel and restarts it.
    max_consecutive_failures
        See :class:`MirroredTunnels` -- controls when a repeatedly failing
        tunnel is given up on instead of retried forever.

    Returns
    -------
    MirroredTunnels
        ``.urls(plate_name)`` gives the mirror URL list to pass as
        ``plate_path`` to ``add_plate`` and friends. Call ``.stop()`` (or
        use as a context manager) when done viewing.
    """
    if local_ports is None:
        local_ports = []
        for _ in range(n_mirrors):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                local_ports.append(probe.getsockname()[1])
    elif len(local_ports) != n_mirrors:
        raise ValueError(f"local_ports has {len(local_ports)} entries but n_mirrors={n_mirrors}.")

    tunnels = MirroredTunnels(
        local_ports=local_ports,
        remote_host=remote_host,
        remote_port=remote_port,
        login_alias=login_alias,
        check_interval=check_interval,
        max_consecutive_failures=max_consecutive_failures,
    )
    tunnels._processes = [tunnels._spawn(port) for port in local_ports]
    tunnels._fail_streaks = [0] * n_mirrors
    tunnels._given_up = [False] * n_mirrors
    tunnels._monitor_thread = threading.Thread(target=tunnels._monitor, daemon=True)
    tunnels._monitor_thread.start()
    print(f"Opened {n_mirrors} independent ssh tunnels to {remote_host}:{remote_port} on local ports {local_ports}.")
    return tunnels


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
    mirror_urls: Optional[List[str]] = None,
    max_workers: int = 8,
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
    mirror_urls
        Several base URLs for the same plate, each reached through its own
        separate ssh tunnel (see :func:`open_mirrored_tunnels`) -- when
        given (and ``open_containers`` is ``None``), each well opens
        through one of these instead of through ``plate``, round-robined
        by :func:`_well_mirror_map`. ``None`` (the default) opens every
        well through ``plate`` itself, same as before.
    max_workers
        Open/check up to this many wells concurrently, across threads --
        see :func:`_map_concurrently`. ``1`` opens them one at a time, in
        ``well_paths`` order.

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

    well_mirror = _well_mirror_map(all_well_paths, mirror_urls) if mirror_urls else None

    def _open_one(well_path: str) -> Optional[OmeZarrContainer]:
        if open_containers is not None:
            container = open_containers.get(well_path)
            if container is None:
                return None
        elif well_mirror is not None:
            try:
                container = open_ome_zarr_container(f"{well_mirror[well_path]}/{well_path}/{kind}")
            except (ValueError, NgioFileNotFoundError, FileNotFoundError):
                return None
        else:
            row, column = well_path.split("/")
            try:
                container = plate.get_image(row, column, kind)
            except ValueError:
                return None
        if label_name is not None and label_name not in container.list_labels():
            return None
        return container

    results = _map_concurrently(_open_one, well_paths, max_workers=max_workers)
    return {well_path: container for well_path, container in zip(well_paths, results) if container is not None}


def build_plate_pyramid(
    plate_path: Union[str, Path, List[Union[str, Path]]],
    kind: Literal["stack", "mip"] = "mip",
    label_name: Optional[str] = None,
    gap_fraction: float = 0.05,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
    max_workers: int = 8,
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
        Full path to the plate's .zarr store -- or several mirror URLs for
        the same store (see :func:`open_mirrored_tunnels`), each reached
        through its own separate ssh tunnel, to spread per-well reads
        across more than one connection.
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
    max_workers
        See :func:`_discover_wells_with_label` -- also controls how many
        wells' arrays are fetched concurrently, per pyramid level, below.

    Returns
    -------
    List[dask.array.Array]
        One array per pyramid level, finest first.

    Raises
    ------
    ValueError
        If no well has the requested image/label.
    """
    mirror_urls = _normalize_mirror_urls(plate_path)
    if plate is None:
        plate = open_ome_zarr_plate(store=mirror_urls[0], mode="r")
    containers = _discover_wells_with_label(
        plate, kind, label_name, open_containers=open_containers, mirror_urls=mirror_urls, max_workers=max_workers
    )

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
        well_order = list(containers.keys())
        arrays = _map_concurrently(
            lambda well_path: _get_array(containers[well_path], level_path, well_offsets[well_path]),
            well_order,
            max_workers=max_workers,
        )
        for well_path, arr in zip(well_order, arrays):
            row_idx, col_idx = row_col_index[well_path]
            y0, x0 = row_idx * pitch_h, col_idx * pitch_w
            canvas[..., y0 : y0 + tile_h, x0 : x0 + tile_w] = arr
        pyramid.append(canvas)

    return pyramid


def _read_plate_wide_features(
    plate_path: Union[str, Path, List[Union[str, Path]]],
    label_name: str,
    kind: Literal["stack", "mip"] = "mip",
    wells: Optional[Union[str, List[str]]] = None,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
    feature_names: Optional[List[str]] = None,
    max_workers: int = 8,
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
        Full path to the plate's .zarr store -- or several mirror URLs for
        the same store (see :func:`open_mirrored_tunnels`); see
        :func:`build_plate_pyramid`'s own ``plate_path`` for details.
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
    max_workers
        See :func:`_discover_wells_with_label` -- also controls how many
        wells' tables are read concurrently here.

    Returns
    -------
    Optional[pandas.DataFrame]
        ``None`` if no (selected) well has a matching features table.
    """
    mirror_urls = _normalize_mirror_urls(plate_path)
    if plate is None:
        plate = open_ome_zarr_plate(store=mirror_urls[0], mode="r")
    containers = _discover_wells_with_label(
        plate,
        kind,
        label_name,
        wells,
        open_containers=open_containers,
        mirror_urls=mirror_urls,
        max_workers=max_workers,
    )

    def _read_one(well_path: str) -> Optional[pd.DataFrame]:
        container = containers[well_path]
        table_name = f"{label_name}_features"
        if table_name not in container.list_tables():
            return None
        row, column = well_path.split("/")
        df = container.get_feature_table(table_name).dataframe.reset_index()
        if "global_id_numeric" in df.columns:
            df["label"] = df["global_id_numeric"]
        else:
            df["label"] = df["label"] + well_label_offset(plate.rows.index(row), plate.columns.index(column))
        if feature_names is not None:
            keep = ["label"] + [c for c in feature_names if c in df.columns and c != "label"]
            df = df[keep]
        return df

    results = _map_concurrently(_read_one, list(containers.keys()), max_workers=max_workers)
    feature_frames = [df for df in results if df is not None]

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
    plate_path: Union[str, Path, List[Union[str, Path]]],
    label_name: str,
    feature_name: str,
    kind: Literal["stack", "mip"] = "mip",
    wells: Optional[Union[str, List[str]]] = None,
    plate: Optional[OmeZarrPlate] = None,
    open_containers: Optional[Dict[str, OmeZarrContainer]] = None,
    max_workers: int = 8,
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
        Full path to the plate's .zarr store -- or several mirror URLs for
        the same store (see :func:`open_mirrored_tunnels`); see
        :func:`build_plate_pyramid`'s own ``plate_path`` for details. Each
        well's raw-zarr read below goes through whichever mirror
        :func:`_well_mirror_map` assigns it, matching whichever mirror
        actually opened its container.
    label_name
        Which label's own measurements to read (``f"{label_name}_features"``).
    feature_name
        The single measurement column to read.
    kind
        "stack" or "mip".
    wells, plate, open_containers
        See :func:`_read_plate_wide_features` -- identical meaning.
    max_workers
        See :func:`_discover_wells_with_label` -- also controls how many
        wells' single-column reads happen concurrently here.

    Returns
    -------
    Optional[pandas.DataFrame]
        Two columns, ``"label"`` (plate-wide-unique) and ``feature_name``
        -- or ``None`` if no (selected) well has a matching features table
        with this column.
    """
    mirror_urls = _normalize_mirror_urls(plate_path)
    if plate is None:
        plate = open_ome_zarr_plate(store=mirror_urls[0], mode="r")
    containers = _discover_wells_with_label(
        plate,
        kind,
        label_name,
        wells,
        open_containers=open_containers,
        mirror_urls=mirror_urls,
        max_workers=max_workers,
    )
    table_name = f"{label_name}_features"
    well_mirror = _well_mirror_map(plate.wells_paths(), mirror_urls)

    def _read_one(well_path: str) -> Optional[pd.DataFrame]:
        container = containers[well_path]
        if table_name not in container.list_tables():
            return None

        table_group_path = f"{well_mirror[well_path]}/{well_path}/{kind}/tables/{table_name}"
        df = _read_one_feature_column_raw(table_group_path, feature_name)
        if df is None:
            full_df = container.get_feature_table(table_name).dataframe.reset_index()
            if feature_name not in full_df.columns:
                return None
            keep = ["label", feature_name] + (["global_id_numeric"] if "global_id_numeric" in full_df.columns else [])
            df = full_df[keep].copy()

        row, column = well_path.split("/")
        if "global_id_numeric" in df.columns:
            df["label"] = df["global_id_numeric"]
        else:
            df["label"] = df["label"] + well_label_offset(plate.rows.index(row), plate.columns.index(column))
        return df[["label", feature_name]]

    results = _map_concurrently(_read_one, list(containers.keys()), max_workers=max_workers)
    feature_frames = [df for df in results if df is not None]

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


def _atomic_add_image_with_retry(
    plate: OmeZarrPlate,
    row: str,
    column: int,
    image_path: str,
    max_attempts: int = 5,
    initial_backoff: float = 1.0,
) -> str:
    """Call ``plate.atomic_add_image``, retrying with exponential backoff
    (plus jitter) if it fails because the plate's own lock couldn't be
    acquired.

    ``atomic_add_image``'s internal lock has a fixed 10-second timeout
    inside ``ngio`` itself (``ngio/utils/_zarr_utils.py``'s
    ``FileLock(_lock_path, timeout=10)`` -- no parameter blimp or any other
    caller can override). A PBS array job registers many wells into the
    same plate concurrently, one per task; under heavy enough contention
    for this single shared lock, a task can exhaust that fixed timeout and
    crash outright rather than simply waiting its turn. Retrying here (ngio
    exposes no retry hook for this lock) gives contention time to clear
    across the whole array job, at the cost of a longer worst-case runtime
    for whichever tasks actually hit it.

    Parameters
    ----------
    plate
        The already-open plate.
    row, column, image_path
        Passed straight through to ``plate.atomic_add_image``.
    max_attempts
        Total attempts before giving up and letting the ``filelock.Timeout``
        propagate.
    initial_backoff
        Seconds to wait before the second attempt; doubles (plus up to 50%
        random jitter, to desynchronize many simultaneously-retrying tasks
        rather than have them all wake up and collide again at once) each
        attempt after that.

    Returns
    -------
    str
        The well's path, as returned by ``atomic_add_image``.
    """
    for attempt in range(1, max_attempts):
        try:
            return plate.atomic_add_image(row=row, column=column, image_path=image_path)
        except filelock.Timeout:
            backoff = initial_backoff * (2 ** (attempt - 1)) * (1 + random.random() * 0.5)
            logger.warning(
                f"Timed out acquiring the plate's own lock while registering well "
                f"{row}{column:02d} (attempt {attempt}/{max_attempts}); retrying in "
                f"{backoff:.1f}s. Expected under heavy concurrent write load (e.g. many "
                "PBS array tasks writing to the same plate at once) -- ngio's own lock "
                "timeout is a fixed 10s, not configurable."
            )
            time.sleep(backoff)
    return plate.atomic_add_image(row=row, column=column, image_path=image_path)


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
    well_relative_path = _atomic_add_image_with_retry(plate, layout.row, layout.column, image_path)

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
