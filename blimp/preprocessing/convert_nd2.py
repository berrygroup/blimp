"""Convert Nikon nd2 files to standard open microscopy environment formats."""
from typing import Union
from pathlib import Path
import os
import glob
import logging

logger = logging.getLogger(__name__)

nd2_to_tiff_jobscript_template = """#!/bin/bash

### Splits nd2 files into standard OME-TIFF format

#PBS -N ConvertND2
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -l walltime=04:00:00
#PBS -o {LOG_DIR}/${{PBS_JOBNAME}}.${{PBS_JOBID}}.out
#PBS -e {LOG_DIR}/${{PBS_JOBNAME}}.${{PBS_JOBID}}.err
#PBS -k oed
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
"""


def find_nd2_files(basepath: Union[Path, str]) -> list:
    """Recursively searches an input directory for.

    .nd2 files and returns a list of the full paths

    Parameters
    ----------
    basepath
        root directory to begin searching

    Returns
    ----------
    list of full paths to nd2 files
    """
    return glob.glob(str(basepath) + "/**/*.nd2", recursive=True)


def generate_pbs_script(
    template: str,
    input_dir: str,
    log_dir: str,
    user: str,
    email: str,
    n_batches: int,
    mip: bool,
    y_direction: str,
) -> str:
    """Formats a PBS jobscript template using input arguments.

    Parameters
    ----------
    template
        PBS jobscript template
    input_dir
        full path to images directory
    log_dir
        full path to where output logs should be written
    user
        usename for job submission (zID on katana)
    email
        email address for notifications
    n_batches
        how many batches into which processing should
        be split
    mip
        whether to save maximum-intensity-projections
    y_direction
        y_direction parameter for nd2_to_ome_tiff

    Returns
    -------
    Template as a formatted string to be written to file
    """
    return template.format(
        INPUT_DIR=input_dir,
        LOG_DIR=log_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        BATCH_MAX=n_batches - 1,
        MIP="-m" if mip else "",
        Y_DIRECTION=y_direction,
    )


def convert_nd2(
    in_path: Union[str, Path],
    job_path: Union[str, Path],
    image_format: str,
    n_batches: int = 1,
    y_direction: str = "down",
    mip: bool = False,
    submit: bool = False,
    user: str = "z1234567",
    email: str = "foo@bar.com",
    dryrun: bool = False,
) -> None:
    """Recursively searches for .nd2 files and creates a job script to convert
    to OME-TIFF using blimp.nd2_to_ome_tiff. Optionally submits the jobs.

    Parameters
    ----------
    in_path
        path to search for .nd2 files
    job_path
        path where the jobscripts should be saved (logs are
        saved in the `log` subdirectory of this path)
    image_format
        "TIFF" or "NGFF" (currently only TIFF implemented)
    n_batches
        number of batches into which the processing should be split.
    mip
        whether to save maximum-intensity-projections
    y_direction
        direction of increasing (stage) y-coordinates (possible
        values are "up" and "down")
    submit
        whether to also submit the batch jobs to the cluster
    user
        username (your zID, must match the path to your data)
    email
        email address for job notifications
    dryrun
        prepare scripts and echo commands without submitting
    """

    if image_format != "TIFF":
        logger.error(f"image_format = {image_format}. Only TIFF format currently implemented")
        os._exit(1)

    # create job/log directory if it does not exist
    job_path = Path(job_path)
    log_path = job_path / "log"
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Jobscripts will be written to {job_path}")

    # search recursively for directories containing nd2 files
    nd2_paths = find_nd2_files(in_path)
    nd2_parent_paths = list({Path(p).parent for p in nd2_paths})

    logger.info(f"Found {len(nd2_parent_paths)} folders countaining {len(nd2_paths)} .nd2 files")
    for i, p in enumerate(nd2_paths):
        logger.debug(f"nd2 file #{i}: {p}")

    job_paths = [job_path / ("batch_convert_nd2_" + str(p.stem) + ".pbs") for p in nd2_parent_paths]

    # create jobscripts using template
    for im_par_path, job_path in zip(nd2_parent_paths, job_paths):
        jobscript = generate_pbs_script(
            template=nd2_to_tiff_jobscript_template,
            input_dir=str(im_par_path),
            log_dir=str(log_path),
            user=user,
            email=email,
            n_batches=int(n_batches),
            mip=mip,
            y_direction=y_direction,
        )
        # write to files
        with open(job_path, "w+") as f:
            f.writelines(jobscript)

    # dryrun
    if dryrun:
        for j in job_paths:
            os.system("echo qsub " + str(j))

    # submit jobs
    if submit:
        for j in job_paths:
            os.system("qsub " + str(j))

    return None
