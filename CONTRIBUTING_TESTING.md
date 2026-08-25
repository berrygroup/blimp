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

**`tox -e lint` currently fails on pre-existing formatting.** As of this writing
this branch has 6 files failing `black==23.1.0` and 2 failing `isort==5.12.0`,
all inherited from `main` (which has 7 and 2 respectively). The CI lint job is
therefore marked `continue-on-error`. To
clear it, run `pre-commit run --all-files`, commit the result as a
formatting-only change, then remove `continue-on-error` from
`.github/workflows/ci.yml`.

**tox builds its own isolated environments.** The conda environment is for
interactive work and direct `pytest` runs; `tox` will create separate virtualenvs
regardless, which is why it appears in both `environment-dev.yml` and the `test`
extra.
