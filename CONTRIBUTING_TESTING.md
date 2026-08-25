# Setting up a test environment

Recipe for a full development install with the test suite working, on macOS or
Linux. Every command below was run end to end before this file was written.

## 1. Clone

```bash
git clone git@github.com:berrygroup/blimp.git
cd blimp
```

Clone rather than downloading a source zip. The version is derived from the git
tags by `setuptools_scm`, so a checkout without `.git` fails to build with
`LookupError: unable to detect version`.

## 2. Create the conda environment

```bash
conda env create -f environment-dev.yml
conda activate blimp-dev
```

This installs Python 3.11 plus the compiled scientific stack (numpy, pandas,
scikit-image, SimpleITK, mahotas) from conda-forge, and the test tooling (tox,
pytest, pytest-cov, pytest-dependency, pre-commit).

`mamba env create -f environment-dev.yml` works identically and solves faster.

## 3. Install blimp in editable mode

```bash
pip install -e ".[test,dev]"
```

This adds blimp itself, plus the pip-only dependencies that conda cannot supply:
`aicsimageio[nd2]`, `cellpose`, `itk-elastix`, `nd2reader`, `pystackreg`,
`welford`.

Two steps rather than one because the dependency set straddles the conda/pip
divide. Do not move the pip-only packages into `environment-dev.yml`: listing a
dependency in both places gives you two copies and imports that resolve to
whichever came last.

## 4. Check the install

```bash
python -c "import blimp; print(blimp.__version__)"
python -c "from aicsimageio.writers import OmeTiffWriter; print('writers OK')"
blimp --help
```

Expected: a version like `0.4.1.dev4+g8c5ef246f` (derived from the git tags, so
it will differ), `writers OK`, and the CLI usage line.

The `aicsimageio.writers` import is worth checking explicitly. It is the one
that breaks if `numcodecs` resolves to 0.16 or newer, because `zarr` 2.x imports
a symbol removed there. Both `requirements.txt` and `environment-dev.yml` pin
`numcodecs<0.16`.

## 5. Run the tests

```bash
tox -e offline
```

Start here: 142 tests, no network access and no reference dataset needed. About
90 s, plus a one-off ~100 s the first time while tox builds its virtualenv.

For the fast inner loop while developing, call pytest directly in the activated
conda environment and skip both the tox virtualenv and the coverage pass:

```bash
pytest tests/ -q -m "not data" --no-cov     # ~20 s
```

```bash
tox
```

Full suite: 213 passed, 2 skipped, about 90 s once the data is present. On the
first run this downloads a 111 MB archive from Figshare which extracts to
`tests/_data` and `tests/_experiments` (about 200 MB). The download is automatic
and is skipped when the dataset is already present.

```bash
tox -e coverage    # coverage report: terminal, XML and HTML
tox -e lint        # pre-commit hooks (see caveat below)
tox -l             # list the default environments
```

`coverage` reports on the accumulated data file, so it only has anything to say
after a test env has run in the same session. It must also run *after*
`covclean`, which erases that file; `envlist` is ordered accordingly. A full
`tox` run measures **65% of `blimp/`** (1488 of 2287 statements, macOS/py311).
Coverage is scoped to the package via `source = blimp` in `[coverage:run]` --
without that, test files score near 100% and inflate the headline to about 80%,
and any stray package tree in the repo root gets measured too.

Note that `tox -l` lists only the `envlist` entries (the py310/311/312 matrix
across linux and macos, plus `lint`, `coverage`, `covclean` and the docs
targets). `offline` is defined but deliberately not in `envlist`, so it does not
appear there even though `tox -e offline` works. `tox list --no-desc` shows
everything.

To run pytest directly, for a single test or with `-k`:

```bash
pytest tests/ -q                          # everything
pytest tests/ -q -m "not data"            # offline subset only
pytest tests/test_quantify_synthetic.py -v
pytest tests/ -k "registration" -v
```

## Environment variables

| Variable | Effect |
| --- | --- |
| `BLIMP_SKIP_TEST_DATA=1` | Skip the dataset download entirely. Data-dependent tests are then skipped rather than failing. Useful offline or on a metered connection. Read by `tests/fetch_test_data.py`, which `tox` runs before the suite. |
| `BLIMP_DOWNLOAD_TEST_DATA=1` | Opt in to downloading the dataset when running `pytest` directly. Equivalent to passing `--download-test-data`. `tox` sets this for you, so it is only needed outside tox. |

To force a fresh download, delete `tests/_data` (or `tests/_data.zip` to
re-fetch rather than re-extract).

## Known caveats

**Two tests skip if `~/.cellpose` is not writable.** `test_segment.py` needs to
cache pretrained cellpose weights. The skip message names the cause; it is not a
failure.

## Install the hooks

The hooks are declared in `.pre-commit-config.yaml` but **git does not run them
until you install them into your clone**:

```bash
pre-commit install
```

Without this, `.git/hooks/pre-commit` does not exist and commits are never
checked. `pre-commit` is in `environment-dev.yml`, so it is already available
once the conda environment is active. This is per-clone and is not carried by
`git clone`, so each person has to do it once.

To check without committing:

```bash
pre-commit run --all-files          # every hook, every file
pre-commit run black --all-files    # one hook
tox -e lint                         # same hooks, in an isolated environment
```

**`tox -e lint` still fails, but no longer on formatting.** `black` and `isort`
now pass across the whole tree. Three hooks fail, all pre-existing and none
fixable by reformatting, so the CI lint job stays `continue-on-error`:

| hook | failures | why it needs a human |
| --- | --- | --- |
| `mypy==1.7.1` | 5 errors | 4 are indexed assignment on an `Optional` in `segment.py`; 1 is a `Path` assigned to a `str`-annotated variable in `convert_nd2.py` |
| `autoflake` | 10 files | unused imports, mostly `import os` -- the hook runs `--remove-all-unused-imports`, i.e. it deletes code |
| `pyupgrade` | 1 file | |

Fix those, then remove `continue-on-error` from `.github/workflows/ci.yml`.

### A note on the pinned versions

Every `rev:` in `.pre-commit-config.yaml` is an exact git tag -- that is how
pre-commit works, there is no floating option. It builds an isolated environment
per hook at that tag, which is what makes the checks reproducible across
machines. So the pins are not a special decision; the question is when to bump
them.

They were last bumped in `ba4d35a` (Nov 2023). As of Aug 2026 that leaves
`black` 27 releases behind, `isort` 9, `mypy` 29, `pyupgrade` 34.

Bumping is a deliberate, separate change, because a formatter upgrade rewrites
files:

| black | files it reformats | non-blank lines changed |
| --- | --- | --- |
| `23.1.0` (pinned) | 6 -> now 0 | 420 |
| `24.10.0` | 21 | 432 |
| `26.5.1` | 22 | -- |

The jump at black 24 is almost entirely cosmetic: 167 of the changed lines are
blank-line insertions, because black 24 made "blank line after the module
docstring" a stable style rule. `isort` is unaffected -- 5.12.0, 5.13.2 and
6.0.1 all report the same 2 files.

If you do bump, do it as `pre-commit autoupdate` plus one formatting-only
commit, separate from any behaviour change, so a later `git bisect` is not
wading through reformatting. Consider holding `mypy` back: 2.x is much stricter
and may surface type errors that need real fixes rather than reformatting.

**The `black`/`isort` line lengths look inconsistent but are not.** `black` is
set to 120 and `isort` to 88 in `pyproject.toml`. Because `isort` runs with
`include_trailing_comma = true`, a wrapped import gets a trailing comma, which
triggers black's magic trailing comma and keeps the import exploded rather than
joining it. Verified stable: applied alternately to a 115-character import they
converge after one pass and do not fight.

**tox builds its own isolated environments.** The conda environment is for
interactive work and direct `pytest` runs; `tox` will create separate virtualenvs
regardless, which is why it appears in both `environment-dev.yml` and the `test`
extra.
