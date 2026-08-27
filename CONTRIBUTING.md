# Contributing guide

## Contributing to BLIMP

Clone BLIMP from source. Clone rather than downloading a source zip: the
version comes from the git tags via `setuptools_scm`, and a checkout
without `.git` fails with `LookupError: unable to detect version`.

    git clone https://github.com/berrygroup/blimp
    cd blimp

Create the development environment, which installs the compiled
scientific stack (numpy, pandas, scikit-image, SimpleITK, mahotas) from
conda-forge along with `tox`, `pytest` and `pre-commit`:

    conda env create -f environment-dev.yml    # or: mamba env create -f ...
    conda activate blimp-dev

Then install blimp itself, plus the dependencies conda cannot supply
(`aicsimageio[nd2]`, `cellpose`, `itk-elastix`, `nd2reader`,
`pystackreg`, `welford`):

    pip install -e '.[dev,test]'

Two steps because the dependencies straddle conda and pip. Do not move
the pip-only packages into `environment-dev.yml`: listing a dependency in
both places installs it twice, and imports resolve to whichever came
last.

Check the install:

    python -c "import blimp; print(blimp.__version__)"
    python -c "from aicsimageio.writers import OmeTiffWriter; print('writers OK')"
    blimp --help

The `writers` import is the one that breaks if `numcodecs` resolves to
0.16 or newer.

Finally, install the pre-commit hooks. They are declared in
`.pre-commit-config.yaml`, but git does not run them until they are
installed into your clone, and `git clone` does not carry them:

    pre-commit install

This is highly recommended, since it will help you to pass the linting
step (see [Code style guide](#code-style-guide)). If you are unable to
decipher some flags, you can still commit using `--no-verify`.

## Branching model

`main` is the release branch and `dev` is the integration branch. Work is
merged into `dev` first and promoted to `main` as a release, so `dev` is
where features accumulate and get tested together.

    feature branch  ->  dev  ->  main
                        (staging)  (releases)

Open pull requests against `dev`, not `main`. Reserve `main` for release
promotions from `dev`.

CI runs the full matrix on pull requests into `main` and `dev`, and again
on pushes to those branches. The second run is not redundant: a push to
`dev` is a merge commit, and it is the first time the combined state of
two independently-green pull requests is tested together.

Read the Docs builds both branches, so `dev` documentation is published
alongside the released version and selectable from the flyout menu at the
bottom of any docs page. `latest` tracks `main`; the `dev` version tracks
`dev`.

## Code style guide

We rely on `black` and `isort` to do the most of the formatting - both
of them are integrated as pre-commit hooks. We use `mypy` to further
analyze the code; use `# type: ignore[error1,error2]` to ignore specific
errors. (`flake8` is present in `.pre-commit-config.yaml` but commented
out; `# noqa:` comments in the tree are from when it ran.)

To check without committing:

    pre-commit run --all-files          # every hook, every file
    pre-commit run black --all-files    # a single hook
    tox -e lint                         # same hooks, isolated environment

CI runs the same hooks and is enforcing, so a failure there is a real
regression.

## Testing

We use `tox` to automate our testing, as well as linting and
documentation creation. To run the tests, run:

    tox -e offline    # no network and no test data needed
    tox               # full suite; fetches the test data on first run

`envlist` covers `py{310,311,312}` on both `linux` and `macos`, so a
single leg is named in full:

    tox -e py311-macos
    tox -e py311-macos --recreate    # if the environment needs rebuilding

No interpreter needs installing by hand: `tox-uv` downloads a standalone
CPython for any version missing from `PATH`. Do not put a
conda environment's `bin/` on `PATH` to supply one: the build then picks
up that environment's headers and fails compiling `lxml`, which looks
like a Python-version incompatibility and is not.

For a faster inner loop, call `pytest` directly in the activated conda
environment:

    pytest tests/ -q -m "not data" --no-cov      # offline subset
    pytest tests/ -q                             # everything
    pytest tests/test_quantify_synthetic.py -v   # one file
    pytest tests/ -k "registration" -v           # by name

Coverage is written to `tests/coverage/`. `tox -e coverage` reports on
the accumulated data, so a test environment must have run first. `tox -e
covclean` erases that data, which is needed when switching branches
since coverage otherwise accumulates across runs.

A missing test dataset **fails** rather than skips, so that a partial run
is never reported as a full one. `tox -e offline` is the supported way to
run without it, and selects the data-free subset explicitly. Two skips
are legitimate and unaffected: the `~200 MB` download is opt-in outside
`tox` (`--download-test-data`), and the `cellpose` tests in
`test_segment.py` skip when `~/.cellpose` is not writable.

Test data is stored on
[figshare](https://figshare.com/articles/dataset/blimp_test_data/23972244).
To add more examples and for distributed testing, download this archive,
add additional subfolders, to the `_data` or `_experiments` folders,
compress:

    zip -r _data.zip _data _experiments

and upload as a new version of the same figshare record
([doi.org/10.6084/m9.figshare.23972244](https://doi.org/10.6084/m9.figshare.23972244))

## Writing documentation

We use `numpy`-style docstrings for the documentation.

In order to build the documentation, run:

    tox -e docs

To validate the links inside the documentation, run:

    tox -e check-docs

If you need to clean the artifacts from previous documentation builds,
run:

    tox -e clean-docs

Run `clean-docs` after renaming or deleting a module. `docs/api/` is
`autosummary` output, and `sphinx-build -M clean` does not prune it, so a
stale `.rst` file is left behind and fails `check-docs`.

The docs are built on Python 3.12, unlike the test matrix. See the
comment on `basepython` in `tox.ini` for why. `tox-uv` fetches it if
needed.

CI builds the docs with `-W`, so any Sphinx warning fails a pull request.
Read the Docs publishes them and does not fail on warnings. Two warnings
you may see locally:

- "failed to reach any of the inventories": `intersphinx` could not reach
  the network. This is environmental rather than a docs fault. Set
  `BLIMP_DOCS_OFFLINE=1` to skip those fetches.
- "Bullet list ends without a blank line": a bullet continuation line is
  not indented under the bullet text. This applies to the argparse
  `description` strings in `blimp/cli/main.py` as well as to docstrings,
  since `sphinx-argparse` renders them as RST.

To build documentation for ipython notebooks, you will also have to
install [pandoc](https://pandoc.org/installing.html), for example with
conda. There are currently no notebooks under `docs/`, so this is only
needed if you add one:

    conda install -c conda-forge pandoc

## Environment variables

| Variable | Effect |
| --- | --- |
| `BLIMP_DOWNLOAD_TEST_DATA=1` | Download the test dataset when running `pytest` directly. Equivalent to `--download-test-data`. Not needed under `tox`. |
| `BLIMP_DOCS_OFFLINE=1` | Empty `intersphinx_mapping`, so `tox -e docs` makes no external requests. Set by CI. |

## Known issues

- The `py312` leg emits ~430,000 NumPy deprecation warnings, almost all
  from `tests/test_quantify.py`. The deprecated call is in `mahotas`
  rather than in blimp, but it becomes an error in a future NumPy.
- The Codecov badge reads `unknown` because no upload has reached Codecov
  yet. Finishing the setup needs the `berrygroup` org authorised at
  codecov.io and the upload token stored as the `CODECOV_TOKEN`
  repository secret. The badge only populates from a default-branch
  upload.
- The `rev:` entries in `.pre-commit-config.yaml` were last updated in
  November 2023. Update them with `pre-commit autoupdate` as its own
  commit, separate from any behaviour change, since a formatter upgrade
  rewrites files. Consider holding `mypy` back, as 2.x is stricter and
  surfaces errors needing real fixes rather than reformatting.
