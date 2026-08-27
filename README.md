# blimp

[![CI](https://github.com/berrygroup/blimp/actions/workflows/ci.yml/badge.svg)](https://github.com/berrygroup/blimp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/berrygroup/blimp/branch/main/graph/badge.svg)](https://codecov.io/gh/berrygroup/blimp)
[![Documentation Status](https://readthedocs.org/projects/blimp/badge/?version=latest)](https://blimp.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/508656801.svg)](https://zenodo.org/badge/latestdoi/508656801)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**B**erry **L**ab **IM**age **P**rocessing

Python tools and workflows for microscopy image pre-processing, processing and
analysis.

Full documentation on
[Read the Docs](https://blimp.readthedocs.io/en/latest/index.html).

## What it does

**Pre-processing**

- **Convert** — batch conversion of microscope-specific formats (Nikon `nd2`,
  Operetta) to OME-TIFF, extracting metadata to a `pandas` DataFrame.
  Maximum-intensity projection along `z` optionally happens during conversion.
- **Correct** — illumination bias correction, fitted from a set of reference
  images and optionally smoothed.
- **Align** — 2D registration between time-points, imaging cycles or channels,
  using `itk-elastix`.

**Processing**

- **Segment** — nuclear segmentation with Cellpose, watershed-based object
  expansion, secondary object segmentation, and resolution of objects
  straddling multiple parents.
- **Quantify** — intensity, morphology and Haralick texture features per
  object, in 2D or 3D, aggregated into tidy DataFrames.

Batch work is aimed at a PBS cluster: the command line interface generates
jobscripts and can submit them.

## Installation

```bash
git clone https://github.com/berrygroup/blimp.git
cd blimp
pip install -e .
```

Clone rather than downloading a source zip — the version is derived from the
git history by `setuptools_scm`.

This also installs the `blimp` command line interface into the environment's
`bin` directory; run `blimp -h` for usage.

Tested on Python 3.10, 3.11 and 3.12, on Linux and macOS.

## Examples

Worked examples are in the `notebooks` folder. Run `notebooks/0_setup.ipynb`
first — it downloads the
[example dataset](https://doi.org/10.6084/m9.figshare.21944927) and writes a
local configuration.

## Contributing

Development environment, test suite and code style are described in the
[contributing guide](CONTRIBUTING.md). In short:

```bash
pip install -e '.[dev,test]'
tox -e offline    # tests needing no network or reference data
tox               # full suite; downloads the reference dataset
tox -e lint       # style checks
tox -e docs       # build the documentation
```
