#!/bin/bash

### Splits a set of operetta images into batches and converts to OME-TIFF

#PBS -N ConvertOperetta
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -k n
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
conda activate berrylab-py310

cd $PBS_O_WORKDIR

python /srv/scratch/{USER}/src/blimp/blimp/preprocessing/operetta_to_ome_tiff.py \
-i $INPUT_DIR/Images -o $OUTPUT_DIR -f Index.idx.xml --batch {N_BATCHES} \
${{PBS_ARRAY_INDEX}} -s -m

conda deactivate
