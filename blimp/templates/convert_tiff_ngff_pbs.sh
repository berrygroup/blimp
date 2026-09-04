#!/bin/bash

### Stitches an existing OME-TIFF pipeline's output (intensity, labels,
### features) into a whole-plate OME-NGFF (OME-Zarr) store

#PBS -N ConvertTiffNGFF
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -l walltime=08:00:00
#PBS -o {LOG_DIR}/
#PBS -e {LOG_DIR}/
#PBS -M {USER_EMAIL}
#PBS -m ae

### The following parameter is modulated at runtime to specify the
### batch number on each node. Batches should run from zero to N_BATCHES-1
### to process all wells

#PBS -J 0-{BATCH_MAX}

###---------------------------------------------------------------------------

INPUT_DIR="{INPUT_DIR}"
PLATE_PATH="{PLATE_PATH}"

source /home/{USER}/.bashrc
conda activate berrylab-py310

cd $PBS_O_WORKDIR

python /srv/scratch/{USER}/src/blimp/blimp/preprocessing/tiff_to_ome_ngff.py convert \
-i "$INPUT_DIR" -o "$PLATE_PATH" --batch {N_BATCHES} ${{PBS_ARRAY_INDEX}} \
-y {Y_DIRECTION} -x {X_DIRECTION} --placement {PLACEMENT} {CHANNEL_NAMES} {LABEL_DIR} {FEATURE_CSV_DIR} {POINT_OBJECT_CHANNEL_NAMES}

conda deactivate
