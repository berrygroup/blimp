"""Convert Perkin-Elmer Operetta image files to standard open microscopy
environment formats."""
import os
import glob
import logging
from pathlib import Path
from typing import Union
from blimp.log import configure_logging

logger = logging.getLogger(__name__)

operetta_to_tiff_jobscript_template = """#!/bin/bash

### Splits a set of operetta images into batches and converts to OME-TIFF

#PBS -N ConvertOperetta
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -l walltime=04:00:00
#PBS -j oe
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

python /srv/scratch/{USER}/src/blimp/blimp/preprocessing/operetta_to_ome_tiff.py -i $INPUT_DIR/Images -o $OUTPUT_DIR -f Index.idx.xml --batch {N_BATCHES} ${{PBS_ARRAY_INDEX}} -s -m

conda deactivate
"""


def find_images_dirs(basepath):
    return glob.glob(basepath + "/**/Images", recursive=True)


def generate_pbs_script(template, input_dir, user, email, n_batches):
    return template.format(
        INPUT_DIR=input_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        BATCH_MAX=n_batches - 1,
    )


def convert_operetta(
    in_path: Union[str, Path],
    job_path: Union[str, Path],
    image_format: str,
    n_batches: int = 1,
    save_metadata_files: bool = True,
    mip: bool = False,
    submit: bool = False,
    user: str = "z1234567",
    email: str = "foo@bar.com",
    dryrun: bool = False,
) -> None:
    """Recursively searches for 'Images' directories and creates a job script
    to submit a batch of operetta-generated TIFFs for export in the specified
    format. Optionally submits the jobs.

    Parameters
    ----------
    in_path
        path to the "Images" directory (including "Images")
    out_path
        path where the converted image data should be saved
    job_path
        path where the jobscripts should be saved
    metadata_file
        name of the xml metadata file inside "Images"
    image_format
        "TIFF" or "NGFF" (currently only TIFF implemented)
    n_batches
        number of batches into which the processing should be split.
    batch_id
        current batch to process
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    mip
        whether to save maximum-intensity-projections
    submit
        whether to also submit the batch jobs to the cluster
    user
        username (your zID, must match the path to your data)
    email
        email address for job notifications
    dryrun
        prepare scripts and echo commands without submitting
    """

    # create jobdir if it does not exist
    job_path = Path(job_path)
    if not job_path.exists():
        job_path.mkdir(parents=True, exist_ok=True)

    # search recursively for "Images" directories
    images_paths = find_images_dirs(in_path)
    images_parent_paths = [Path(p).parent for p in images_paths]
    jobscript_paths = [
        job_path / ("batch_convert_" + str(p.stem) + ".pbs")
        for p in images_parent_paths
    ]

    # create jobscripts using template
    for images_parent_path, jobscript_path in zip(images_parent_paths, jobscript_paths):
        jobscript = generate_pbs_script(
            operetta_to_tiff_jobscript_template,
            images_parent_path,
            user,
            email,
            int(n_batches),
        )
        # write to files
        with open(jobscript_path, "w+") as f:
            f.writelines(jobscript)

    # dryrun
    if dryrun:
        for j in jobscript_paths:
            os.system("echo qsub " + str(j))

    # submit jobs
    if submit:
        for j in jobscript_paths:
            os.system("qsub " + str(j))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="convert_operetta")

    parser.add_argument(
        "-i", "--in_path", help="directory containing the input files", required=True
    )

    parser.add_argument(
        "-j",
        "--job_path",
        default=os.getcwd(),
        help="directory to write the jobscripts",
        required=True,
    )

    parser.add_argument(
        "--image_format",
        default="TIFF",
        help="output format for images (TIFF or NGFF, currently only TIFF implemented)",
    )

    parser.add_argument(
        "--batch",
        default=1,
        help="if files are processed as batches, provide the number of batches",
    )

    parser.add_argument(
        "-m",
        "--mip",
        default=False,
        action="store_true",
        help="whether to save maximum intensity projections",
    )

    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="whether to submit jobs after creating jobscripts",
    )

    parser.add_argument(
        "-n",
        "--dryrun",
        default=False,
        action="store_true",
        help="Dry-run mode. Echo submission commands without submitting",
    )

    parser.add_argument("--user", help="Your zID", required=True)

    parser.add_argument(
        "--email", help="Email address for job notifications", required=False
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (e.g. -vvv)",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    convert_operetta(
        in_path=args.in_path,
        job_path=args.job_path,
        image_format=args.image_format,
        n_batches=args.batch,
        mip=args.mip,
        submit=args.submit,
        user=args.user,
        email=args.email,
        dryrun=args.dryrun,
    )
