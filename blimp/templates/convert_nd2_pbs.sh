#!/bin/bash

### Splits nd2 files into standard OME-TIFF format

#PBS -N ConvertND2
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -l walltime=08:00:00
#PBS -o {LOG_DIR}/
#PBS -e {LOG_DIR}/
#PBS -M {USER_EMAIL}
#PBS -m ae

### The following parameter is modulated at runtime to specify the
### batch number on each node. Batches should run from zero to N_BATCHES-1
### to process all files

#PBS -J 0-{BATCH_MAX}

###---------------------------------------------------------------------------

INPUT_DIR={INPUT_DIR}
OUTPUT_DIR={INPUT_DIR}/OME-TIFF

source /home/{USER}/.bashrc
conda activate berrylab-default

cd $PBS_O_WORKDIR

python /srv/scratch/{USER}/src/blimp/blimp/preprocessing/nd2_to_ome_tiff.py \
-i $INPUT_DIR -o $OUTPUT_DIR --batch {N_BATCHES} ${{PBS_ARRAY_INDEX}} \
{MIP} -y {Y_DIRECTION}

conda deactivate
