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

Start here: 167 tests, no network access and no reference dataset needed. About
90 s, plus a one-off ~100 s the first time while tox builds its virtualenv.

This is the only supported way to run without the dataset -- a missing dataset
otherwise fails the run. See [Why a missing dataset
fails](#why-a-missing-dataset-fails).

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

## Testing against multiple Python versions

`envlist` covers `py{310,311,312}` on both linux and macos. Nothing needs to be
installed by hand:

```bash
pip install tox
tox -e py310-macos    # or py311-macos, py312-macos
```

tox reads `requires = tox-uv` from `tox.ini` and provisions the plugin into
`.tox/.tox/` on first use; uv then downloads any missing interpreter as a
standalone CPython build. The system Python, conda, and your shell are all left
alone. Verified from a clean state: with `tox-uv` absent and no `python3.10`
anywhere on `PATH`, `tox -e py310-macos` provisioned the plugin, fetched 3.10.21
and passed all 238 tests.

Two settings make this reliable, both deliberate:

- **`skip_missing_interpreters = false`.** With `true` — the previous value — a
  leg whose interpreter is missing reports `SKIP`, and a whole-envlist `tox` run
  **still exits 0**. A machine with one Python reported success having run one
  leg of six, which is how 3.10 and 3.12 went untested. Now a missing
  interpreter is a `FAIL`. This does not affect the platform guard: the `-linux`
  legs still skip cleanly on macOS, and a full `tox` still exits 0 with only
  those skipping.
- **`runner = uv-venv-runner`** in `[testenv]`. Builds an environment in ~50 s
  rather than ~250 s, with an identical result (238 passed either way). Not
  `uv-venv-lock-runner`, which runs `uv sync --locked` and fails without a
  `uv.lock`.

### Don't borrow a conda env for this

Putting a conda env's `bin/` on `PATH` does let tox discover the interpreter,
but the resulting build is contaminated by that env's headers. Concretely,
`PATH="$HOME/anaconda3/envs/py310/bin:$PATH" tox -e py310-macos` fails compiling
`lxml`, because it picks up `envs/py310/include/libxml2` and the vendored
`libxml2` there is not the one `lxml` expects. The same leg passes against a
clean uv-provided 3.10. The failure looks like a Python-version incompatibility
and is not one, so it is worth recognising.

`pyenv` also works but compiles from source and wants shell integration.

### What CI covers

`.github/workflows/ci.yml` runs `ubuntu-latest` and `macos-latest` × Python
3.10, 3.11, 3.12 with `fail-fast: false`, so Linux coverage comes free on every
push.

CI invokes the fully-qualified env (`tox -e py310-linux` and so on) rather than
the factor-less `tox -e py`. `-e py` runs on whatever interpreter invoked tox
and drops the `platform` guard, which meant the `-linux`/`-macos` envs were
never actually exercised in CI — the OS factor is mapped from the runner in the
matrix `include:` block.

CI installs `tox-uv` explicitly and does **not** download interpreters: uv is
invoked with `--python-preference system`, so each leg reuses the interpreter
`actions/setup-python` provides. Verified — no download occurs for a version
already on `PATH`.

Note that the package declares no `python_requires`, so nothing but this matrix
records which versions are supported.

## Coverage and Codecov

Every matrix leg writes `tests/coverage/coverage.xml` and CI uploads it with a
per-leg flag (`Linux-py3.10`, `macOS-py3.12`, ...), so coverage is attributed by
OS and version rather than merged blindly.

Locally:

```bash
tox -e py311-macos              # writes tests/coverage/coverage.xml
tox -e coverage                 # combined terminal + XML + HTML report
open tests/coverage/html/index.html
```

`tox -e coverage` `depends` on the matrix legs, so a full `tox` run reports the
combined figure across every leg that ran. `tox -e covclean` erases the data
first — worth doing when switching branches, since `--cov-append` otherwise
accumulates across runs.

Two details that are easy to get wrong:

- **`relative_files = True`** in `[coverage:run]`. Without it the XML records
  `<source>/Users/you/src/blimp/blimp</source>`. That absolute path differs on
  every machine and on each CI runner, so Codecov cannot reliably map files onto
  the repo tree and the six uploads do not merge into one report. With it,
  `<source>` is just `blimp`. Both code paths honour it — pytest-cov via
  `--cov-config=tox.ini`, and the standalone `coverage` tool because it
  auto-discovers `[coverage:*]` in `tox.ini`.
- **`codecov.yml` is validated, not guessed.** A malformed file is silently
  ignored, which is indistinguishable from Codecov being broken:
  ```bash
  curl -X POST --data-binary @codecov.yml https://api.codecov.io/validate
  ```

### Codecov account setup

The badge has read `unknown` since it was added, because no upload has ever
reached Codecov (`activated: false`, zero branches known). To finish the setup:

1. Sign in at codecov.io with GitHub and grant access to the `berrygroup` org.
2. Add `blimp`, copy the upload token, and store it as the repository secret
   `CODECOV_TOKEN` (Settings → Secrets and variables → Actions).
3. Merge to `main`. The badge only populates from a default-branch upload, so it
   stays `unknown` until then even if PR runs upload successfully.

The action is `codecov/codecov-action@v5`, which restores tokenless upload for
public repos — so coverage still reports on PRs from forks, where `secrets` is
empty. `fail_ci_if_error` is deliberately `false`: a Codecov outage should not
fail a run whose tests passed. The trade-off is that a broken upload is silent,
so confirm the PR comment appears.

`codecov.yml` sets `after_n_builds: 6`, so Codecov waits for all six legs before
setting a status or commenting. Without it the first leg to finish sets the
status and the number moves as the rest arrive. The `project` status is
enforcing (fails on a >1% drop); the `patch` status is `informational: true`
because a 59% codebase cannot clear an 80% patch target immediately, and a check
that always fails gets ignored. Flip it once new code routinely passes.

## Why a missing dataset fails

73 of the 240 tests are marked `data` and need the reference dataset. If it
cannot be obtained, both the fetch script and the `test_data` fixture fail
rather than skip:

- `tests/fetch_test_data.py` exits non-zero, so `tox` stops at `commands_pre`.
- the `test_data` fixture calls `pytest.fail`, naming the three ways forward.

The alternative -- skipping -- makes a run that exercises 70% of the suite
report the same green as a full one. That is the same failure mode as
`skip_missing_interpreters = true`: a real problem that reads as success. A
Figshare outage is therefore on the critical path for a PR, which is the
intended trade: a blocked PR is recoverable, a false pass is not.

To run without the dataset, say so explicitly with `tox -e offline`. Two skips
remain legitimate and are unaffected: `--download-test-data` gates the ~200 MB
download so no test run pulls it silently, and the cellpose tests skip when the
model cache under `~/.cellpose` is not writable, which is an environment fault
rather than missing reference data.

## Documentation environments

| Command | Does |
| --- | --- |
| `tox -e docs` | Builds the HTML docs into `docs/_build/html`. |
| `tox -e check-docs` | Runs `linkcheck` with `-W`, so any warning fails the run. |
| `tox -e clean-docs` | Deletes `docs/_build` and `docs/api`. |

`docs/api/` is `autosummary` output and is gitignored. It is **not** pruned by
`sphinx-build -M clean`, which only removes `_build/`, so a renamed or deleted
module leaves stale `.rst` files behind that then fail `check-docs` under `-W`.
`clean-docs` removes both directories directly; run it after renaming a module.

Two things to know if `check-docs` warns:

- `intersphinx` fetches inventories over the network. Without it (offline, or
  behind a proxy) you get one "failed to reach any of the inventories" warning
  per mapping, which `-W` makes fatal. These are environmental, not docs faults.
- Bullet continuation lines must be indented to align under the bullet text.
  This applies to argparse `description` strings in `blimp/cli/main.py` as well
  as docstrings, because `sphinx-argparse` renders them as RST. Unindented
  continuations give "Bullet list ends without a blank line", attributed to a
  line number with no filename, which is tedious to trace.

There is no spelling check. It was configured but never functional: it needs the
enchant C library (pip cannot supply it), and the wordlist and filter module its
config named were absent from the repo. Reinstating it means providing all three.

## Environment variables

| Variable | Effect |
| --- | --- |
| `BLIMP_DOWNLOAD_TEST_DATA=1` | Opt in to downloading the dataset when running `pytest` directly. Equivalent to passing `--download-test-data`. Not needed under `tox`, which fetches the dataset in `commands_pre` before pytest starts. |

There is deliberately no variable for running without the dataset. A missing
dataset fails; `tox -e offline` selects the data-free subset explicitly. See
[Why a missing dataset fails](#why-a-missing-dataset-fails).

To force a fresh download, delete `tests/_data` (or `tests/_data.zip` to
re-fetch rather than re-extract).

## Known caveats

**Two tests skip if `~/.cellpose` is not writable.** `test_segment.py` needs to
cache pretrained cellpose weights. The skip message names the cause; it is not a
failure.

**~430,000 warnings on the py312 leg.** Cosmetic today, a real break later, and
not a Python-version problem despite only appearing on that leg. The legs
resolve different NumPy versions (3.10 → 2.2.6, 3.11 → 2.4.6, 3.12 → 2.5.2), and
NumPy 2.5 deprecated in-place `arr.shape = ...` assignment. The call is in
`mahotas`, not in blimp — nothing in `blimp/` assigns to `.shape` — so it is
upstream's to fix, but it will become an error in a future NumPy and take
`quantify` with it. `tests/test_quantify.py` alone accounts for 425,534 of them.
Worth an upstream check before pinning anything.

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
