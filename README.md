# blimp

![CI](https://github.com/berrygroup/blimp/actions/workflows/ci.yml/badge.svg)
![Tests](tests/coverage/tests.svg)
![Coverage](tests/coverage/coverage.svg)
![Read the docs](https://readthedocs.org/projects/blimp/badge/?version=latest)
![DOI](https://zenodo.org/badge/508656801.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)

**B**erry **L**ab **IM**age **P**rocessing

Main code repository for image pre-processing, processing, and analysis tools and workflows.

## General notes

Modules are written in Python 3, and can be executed within the conda environment for image processing `berrylab-default`, or any other compatible conda env or virtualenv.

Documentation on [Read The Docs](https://blimp.readthedocs.io/en/latest/index.html)

## Installation

```bash
git clone https://github.com/berrygroup/blimp.git
cd blimp
pip install -e .
```

or

```bash
pip install -e '.[dev,test]'
```

for the development tools.

The installation will make many functions externally accessible and will also install the `blimp` command line interface (CLI) in the `bin` directory of the installation environment. This should be accessible in your path.

## Documentation

Documentation is generated directly from code using [Sphinx](https://www.sphinx-doc.org/en/master/) and the [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension. To generate documentation use `tox`,

```bash
tox -e docs
```

The html documentation will then be available at `/docs/_build/html/index.html`.

## Application Programming Interface (API) and Examples

`blimp` is a python package, with the API defined in the documentation. Several examples are provided in the `notebooks` folder. To use these interactively, first run `notebooks/0_setup.ipynb` which will download the example data and create a example-specific configuration.

Example data is stored on [figshare](https://figshare.com/articles/dataset/blimp_example_data/21944927). To add more examples and for distributed testing, download this archive, add additional subfolders, to the `_data` folder, compress:

```bash
zip -r _data.zip _data
```

and upload as a new version of the same figshare record ([doi.org/10.6084/m9.figshare.21944927](https://doi.org/10.6084/m9.figshare.21944927)).

## Command Line Interface (CLI)

To access help type `blimp -h`.

## Pre-processing

- **Convert**: conversion of microscope-specific file formats to a common file format for uniform downstream processing
- **Correct**: correct acquisition artefacts such as illumination biases
- **Align**: registration of images between time-points, imaging cycles, or channel-specific microscope misalignment

### Convert

blimp can convert batches of raw microscope files to common file formats and extract metadata. Image file formats chosen are [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/) and [OME-NGFF](https://ngff.openmicroscopy.org/latest/). Note: OME-NGFF is not yet implemented.

For further analysis after preprocessing, it is recommended to read both formats using the `AICSImage` class from the [aicsimageio](https://github.com/AllenCellModeling/aicsimageio) package, to ensure image layout and metadata are consistently assigned.

For example, to convert Nikon nd2 files:

```bash
blimp -vv convert nd2 -i /path/to/input/dir -j /path/to/write/pbs/jobscripts --user {zID} --submit
```

This will initiate a search of the input directory for the corresponding file-types, then generate PBS jobscripts to call the conversion functions in batch mode. These files can also submit jobscripts. Converted images and metadata will be written as a new subdirectory of the folder containing the images.

In this case PBS scripts includes the following,

```bash
source /home/{USER}/.bashrc
conda activate berrylab-default
```

which depends on conda being correctly setup within `.bashrc` and a functional `berrylab-default` conda env. If this is not the case, then the example PBS template will need to be edited. The location of this template can be provided on the command line.

### Metadata

#### OME-TIFF

During image conversion, metadata such as physical pixel dimensions and channel names are saved together with the image data in the OME-TIFF. Other metadata is contained in the image filename.

#### Dataframe

During image conversion, additional metadata (such as acquisition times, stage position), is extracted, either from accompanying metadata files, or from the microscope-specific image file. This is combined into a `pandas` DataFrame, and saved as a `.csv` and `.pkl` file. Image filenames are written in the dataframe for cross-referencing with image data.

### Projections

It is extremely common to analyse 2D data derived from 3D imaging volumes using maximum-intensity projection along the axis of the objective lens (z-axis). `blimp convert` can perform maximum intensity projections during conversion. These are saved in `OME-TIFF-MIP` subfolders (along with corresponding metadata). The commandline option `--mip` is used to specify that maximum intensity projections should be performed. Note that original microscope-specific files, as well as conversions of the data containing z-resolution are retained in this case.

### Illumination correction

Not yet implemented

### Image registration

Not yet implemented

## Contributing and Code Style

We have implemented style guide checks using `tox`,

```bash
tox -e lint
```

For further info on formatting and contributing, see the [contributing guide](CONTRIBUTING.md).
