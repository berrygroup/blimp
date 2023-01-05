# blimp

**B**erry **L**ab **IM**age **P**rocessing

Main code repository for image pre-processing, processing, and analysis tools and workflows

## General notes

Modules are written in Python 3, and can be executed within the conda environment for image processing `berrylab-default`, or any other compatible conda env or virtualenv.

Command-line interfaces use the `argparse` library, so usage can be accessed using the `-h` flag (e.g. `python convert_operetta.py -h`).

## Pre-processing

Pre-processing takes microscope-specific file formats and converts then to a common file format for uniform downstream processing. For this package, the file formats chosen for images are [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/) and [OME-NGFF](https://ngff.openmicroscopy.org/latest/). However, OME-NGFF is not yet implemented. It is recommended to read these using the `AICSImage` class from the [aicsimageio](https://github.com/AllenCellModeling/aicsimageio) package, to ensure image layout and metadata are consistently assigned. 

Pre-processing also involves correction of acquisition artefacts such as illumination biases and image alignment (either from multiple time-points, imaging cycles, or from channel-specific misalignment of microscopes).

### Image file format conversion and metadata extraction

To convert between file formats, this can be acheived for a single input directory using `nd2_to_ome_tiff.py` or `operetta_to_ome_tiff.py` in the following way,

```
python blimp/preprocessing/nd2_to_ome_tiff.py -i /path/to/input/dir -o /path/to/output/dir
python blimp/preprocessing/operetta_to_ome_tiff.py -i /path/to/input/dir -o /path/to/output/dir
```

However, it is more common to process files in batches using HPC. To facilitate this on a PBS system (such as katana), these functions accept the `--batch` argument. In this case, the main entry points are `convert_nd2.py` and `convert_operetta.py`, which search the input directories for specific file-types, then generate PBS jobscripts to call the corresopnding conversion functions in batch mode. These files can also submit the jobscripts. These `convert` tools can be executed in the following way,

```
python blimp/preprocessing/convert_nd2.py -i /path/to/imput/dir -o /path/to/output/dir -j /path/to/write/pbs/jobscripts --submit
python blimp/preprocessing/convert_operetta.py -i /path/to/imput/dir -o /path/to/output/dir -j /path/to/write/pbs/jobscripts --submit

```

In this case PBS scripts use the following,
```
source /home/{USER}/.bashrc
conda activate berrylab-default
```
which depends on conda being correctly setup within `.bashrc` and a functional `berrylab-default` conda env.

Alternatives using virtualenv are of course possible, however this require changes to source code.

#### Metadata

##### OME-TIFF
During image conversion, metadata such as physical pixel dimensions and channel names are saved together with the image data in the OME-TIFF. Other metadata is contained in the image filename.

##### Dataframe
During image conversion, additional metadata (such as acquisition times, stage position), is extracted, either from accompanying metadata files, or from the microscope-specific image file. This is combined into a `pandas` DataFrame, and saved as a `.csv` and `.pkl` file. Image filenames are found in the dataframe for cross-referencing with image data.

#### Projections

It is extremely common to analyse 2D data derived from 3D imaging volumes using maximum-intensity projection along the axis of the objective lens (z-axis). Both `nd2_to_ome_tiff.py` or `operetta_to_ome_tiff.py` can perform maximum intensity projections during conversion. These are saved in `OME-TIFF-MIP` subfolders (along with corresponding metadata). The commandline option `--mip` is used to specify that maximum intensity projections should be performed. Note that original microscope-specific files, as well as conversions of the data containing z-resolution are retained in this case.

### Illumination correction

Not yet implemented

### Image registration

Not yet implemented