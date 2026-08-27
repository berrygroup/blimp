#!/usr/bin/env python
"""Ensure the reference test dataset is present.

Run automatically by ``tox`` (``commands_pre``) before the test suite, and
usable standalone::

    python tests/fetch_test_data.py

The download is skipped when the dataset is already extracted, so repeat runs
cost nothing.

Exits non-zero if the dataset cannot be obtained. This is deliberate: the
dataset is required by 73 of 240 tests, and a run that quietly omits them is
indistinguishable from a passing run. To run only the tests that need no
dataset, ask for that explicitly with ``tox -e offline``.
"""
import sys
import pathlib

# Allow running from an uninstalled checkout (tox installs the package, but a
# bare `python tests/fetch_test_data.py` should work too).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    try:
        from blimp.data import load_test_data
    except ImportError as exc:
        print(f"Could not import blimp: {exc}", file=sys.stderr)
        print("Install the package first (`pip install -e .`), or run via `tox`.", file=sys.stderr)
        return 1

    try:
        path = load_test_data()
    except Exception as exc:
        print(f"Reference dataset download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "The dataset is required by the data-marked tests. Run `tox -e offline`"
            " to run only the tests that do not need it.",
            file=sys.stderr,
        )
        return 1

    print(f"Reference dataset available at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
