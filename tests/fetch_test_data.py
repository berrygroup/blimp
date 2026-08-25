#!/usr/bin/env python
"""Ensure the reference test dataset is present.

Run automatically by ``tox`` (``commands_pre``) before the test suite, and
usable standalone::

    python tests/fetch_test_data.py

The download is skipped when the dataset is already extracted, so repeat runs
cost nothing. Set ``BLIMP_SKIP_TEST_DATA=1`` to skip the fetch entirely; the
data-dependent tests will then skip rather than fail, which is the right
behaviour for an offline or air-gapped machine.

Exits 0 even when the download fails, so that a network outage degrades the
run to the offline subset instead of breaking the whole build.
"""
import os
import sys
import pathlib

# The dataset ships as a zip of ~200 MB from Figshare.
_SKIP_ENV = "BLIMP_SKIP_TEST_DATA"

# Allow running from an uninstalled checkout (tox installs the package, but a
# bare `python tests/fetch_test_data.py` should work too).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    if os.environ.get(_SKIP_ENV, "").strip() not in ("", "0", "false", "False"):
        print(f"{_SKIP_ENV} is set: skipping reference dataset download.")
        print("Data-dependent tests will be skipped. Run with -m 'not data' to select them explicitly.")
        return 0

    # Imported here so that --help and the skip path work without the package
    # being importable.
    try:
        from blimp.data import load_test_data
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"Could not import blimp ({exc}); skipping dataset download.", file=sys.stderr)
        return 0

    try:
        path = load_test_data()
    except Exception as exc:
        # A failed download must not break the build: the suite degrades to the
        # offline subset.
        print(f"Reference dataset download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Continuing without it -- data-dependent tests will be skipped.", file=sys.stderr)
        return 0

    print(f"Reference dataset available at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
