"""Shared pytest configuration for the blimp test suite.

Design notes
------------
Tests are split into two classes:

* **unit tests** -- pure functions exercised on small in-memory arrays. These
  must run offline, in any working directory, with no downloaded data.
* **data tests** -- tests that need the reference microscopy dataset. These are
  marked ``@pytest.mark.data`` and are skipped automatically when the dataset
  is not present locally, unless ``--download-test-data`` is passed.

Previously a single ``autouse`` fixture called ``load_test_data()`` (a Figshare
download) before *every* test, so the whole suite -- including pure unit tests
-- failed without network access. The dataset fixture is now opt-in and
session-scoped.
"""
import os
import shutil

import numpy as np
import pytest

# --- markers / options ----------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--download-test-data",
        action="store_true",
        default=False,
        help=(
            "Allow tests to download the reference dataset from Figshare if it is missing. "
            "Equivalent to setting BLIMP_DOWNLOAD_TEST_DATA=1. Under tox the dataset is "
            "fetched by tests/fetch_test_data.py before the suite runs, so this is only "
            "needed for a bare `pytest` invocation."
        ),
    )


# Markers are declared in pyproject.toml under [tool.pytest.ini_options]; with
# --strict-markers there, registering them again here would be redundant.


def _download_permitted(config) -> bool:
    """Downloading is opt-in, via the CLI flag or the environment variable."""
    if config.getoption("--download-test-data"):
        return True
    return os.environ.get("BLIMP_DOWNLOAD_TEST_DATA", "").strip() not in ("", "0", "false", "False")


# --- determinism ----------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_global_rng():
    """Seed the legacy global numpy RNG before every test.

    The existing tests call ``np.random.rand`` etc. without seeding, which makes
    failures unreproducible: ``test_register_2D_preserve_dtype`` registers two
    10x10 pure-noise images with elastix, and on some draws the optimiser fails
    to converge and raises, so the test failed intermittently depending on how
    many random numbers earlier tests had consumed.

    Seeding per test makes each test independent of execution order and of the
    rest of the suite. New tests should prefer an explicit
    ``np.random.default_rng(seed)`` generator over the global state.
    """
    np.random.seed(42)


# --- configuration -------------------------------------------------------


@pytest.fixture(scope="session")
def _experiment_dir(tmp_path_factory):
    """One results directory for the whole session.

    Deliberately session-scoped, not per-test. ``tests/TestData_constants.py``
    computes ``RESULTS_DIR`` from ``blimp_config.EXPERIMENT_DIR`` at *import*
    time, and that module is re-imported by ``get_data_config()``; a per-test
    directory therefore could not be seen consistently by the code under test.
    Keeping it stable for the session makes the paths that tests actually read
    and write agree with each other.
    """
    return str(tmp_path_factory.mktemp("experiments"))


@pytest.fixture(autouse=True)
def blimp_test_config(_experiment_dir, monkeypatch):
    """Point the blimp config at test locations.

    Autouse, but deliberately *offline*: it only sets configuration values and
    never touches the network. Results are written under a temporary directory
    so tests cannot clobber the repository.
    """
    from blimp.constants import SCRIPTS_DIR, blimp_config

    data_dir = os.path.join(SCRIPTS_DIR, "tests", "_data")
    experiment_dir = _experiment_dir

    # Several tests write feature tables to <EXPERIMENT_DIR>/test_results but
    # nothing creates it, so pandas raised
    # "Cannot save file into a non-existent directory". Create it up front.
    os.makedirs(os.path.join(experiment_dir, "test_results"), exist_ok=True)

    monkeypatch.setattr(blimp_config, "EXPERIMENT_DIR", experiment_dir, raising=False)
    monkeypatch.setattr(blimp_config, "BASE_DATA_DIR", data_dir, raising=False)
    blimp_config.add_data_config("TestData", os.path.join(SCRIPTS_DIR, "tests", "TestData_constants.py"))
    return blimp_config


# --- reference dataset ---------------------------------------------------


def _dataset_is_present():
    from blimp.constants import SCRIPTS_DIR

    base = os.path.join(SCRIPTS_DIR, "tests")
    resources = os.path.join(base, "_data", "resources")
    return os.path.isdir(os.path.join(base, "_data")) and os.path.isfile(os.path.join(resources, "affine.txt"))


@pytest.fixture(scope="session")
def test_data(request):
    """Session-scoped access to the reference dataset.

    Extracts a local ``_data.zip`` if present; downloads only when explicitly
    permitted via ``--download-test-data``. Otherwise fails the test -- see the
    message below for why this is not a skip.
    """
    from blimp.constants import SCRIPTS_DIR

    base = os.path.join(SCRIPTS_DIR, "tests")

    if _dataset_is_present():
        return os.path.join(base, "_data")

    archive = os.path.join(base, "_data.zip")
    if os.path.exists(archive):
        shutil.unpack_archive(archive, base)
        if _dataset_is_present():
            return os.path.join(base, "_data")

    if _download_permitted(request.config):
        from blimp.data import load_test_data

        load_test_data()
        if _dataset_is_present():
            return os.path.join(base, "_data")
        pytest.fail("Test data download completed but the dataset is still incomplete.")

    # Deliberately a failure, not a skip: a missing dataset silently removing
    # 73 of 240 tests is indistinguishable from a passing run. To select the
    # data-free subset, say so explicitly with `tox -e offline`.
    pytest.fail(
        "Reference dataset not available locally, so this test cannot run.\n"
        "  - `tox` fetches it automatically before the suite\n"
        "  - `pytest --download-test-data` fetches it now (~200 MB from Figshare)\n"
        "  - `tox -e offline` runs only the tests that need no dataset",
        pytrace=False,
    )


@pytest.fixture()
def _ensure_test_data(test_data):
    """Backwards-compatible alias for the historical fixture name."""
    return test_data


# --- optional heavy models -----------------------------------------------


@pytest.fixture(scope="session")
def cellpose_models():
    """Skip when the cellpose model cache is unavailable.

    cellpose downloads pretrained weights into ``~/.cellpose/models`` on first
    use. On a machine where the home directory is read-only or sandboxed this
    raises PermissionError/FileExistsError from deep inside the library, which
    is an environment problem rather than a blimp defect -- so these tests skip
    with an explanatory message instead of failing the suite.
    """
    cache = os.path.join(os.path.expanduser("~"), ".cellpose", "models")
    try:
        os.makedirs(cache, exist_ok=True)
        probe = os.path.join(cache, ".blimp_write_probe")
        with open(probe, "w") as fh:
            fh.write("")
        os.remove(probe)
    except OSError as exc:
        pytest.skip(f"cellpose model cache is not writable ({exc.__class__.__name__}: {exc}).")
    return cache


def pytest_collection_modifyitems(config, items):
    """Auto-mark anything using the dataset fixtures so ``-m "not data"`` works."""
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "test_data" in fixtures or "_ensure_test_data" in fixtures:
            item.add_marker(pytest.mark.data)
