"""Convert Nikon nd2 files to standard open microscopy environment formats."""
from typing import List, Union
from pathlib import Path
import os
import re
import glob
import logging

from blimp.utils import read_template

logger = logging.getLogger(__name__)


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
    output_dir: str,
    log_dir: str,
    user: str,
    email: str,
    n_batches: int,
    mip: bool,
    keep_stacks: bool,
    y_direction: str,
    channel_names: Union[str, List[str], None] = None,
) -> str:
    """Formats a PBS jobscript template using input arguments.

    Parameters
    ----------
    template
        PBS jobscript template
    input_dir
        full path to images directory
    output_dir
        full path to output directory
    log_dir
        full path to where output logs should be written
    user
        usename (zID on katana) for job submission and location of scripts
    email
        email address for notifications
    n_batches
        how many batches into which processing should
        be split
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    y_direction
        y_direction parameter for nd2_to_ome_tiff
    channel_names
        List of channel names in case those found in the
        image metadata are incorrect and need to be replaced

    Returns
    -------
    Template as a formatted string to be written to file
    """
    if channel_names is None:
        channel_names = ""
    else:
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        channel_names = "--channel_names " + (" ".join(channel_names))

    return template.format(
        INPUT_DIR=input_dir,
        OUTPUT_DIR=output_dir,
        LOG_DIR=log_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        BATCH_MAX=n_batches - 1,
        MIP="--mip" if mip else "",
        KEEP_STACKS="--keep_stacks" if keep_stacks else "",
        Y_DIRECTION=y_direction,
        CHANNEL_NAMES=channel_names,
    )


def convert_nd2(
    in_path: Union[str, Path],
    job_path: Union[str, Path],
    image_format: str,
    template_path: Union[str, Path, None] = None,
    n_batches: int = 1,
    y_direction: str = "down",
    channel_names: Union[str, List[str], None] = None,
    mip: bool = False,
    keep_stacks: bool = True,
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
    template_path
        path to a template for the PBS jobscript
        (default templates/convert_nd2_pbs.sh)
    n_batches
        number of batches into which the processing should be split.
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    y_direction
        direction of increasing (stage) y-coordinates (possible
        values are "up" and "down")
    channel_names
        List of channel names in case those found in the
        image metadata are incorrect and need to be replaced
    submit
        whether to also submit the batch jobs to the cluster
    user
        username (your zID)
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
    logger.info(f"Jobscripts will be written to {job_path.resolve()}")

    # search recursively for directories containing nd2 files
    nd2_paths = find_nd2_files(in_path)
    nd2_parent_paths = list({Path(p).parent for p in nd2_paths})

    logger.info(f"Found {len(nd2_parent_paths)} folders countaining {len(nd2_paths)} .nd2 files")
    for i, p in enumerate(nd2_paths):
        logger.debug(f"nd2 file #{i}: {p}")

    job_paths = [job_path / ("batch_convert_nd2_" + str(p.stem) + ".pbs") for p in nd2_parent_paths]

    # check that zID of input path matches user's zID (otherwise no write access for output)
    out_paths = []
    # find the zID in the path
    pattern = re.compile(r"/z\d{7}")
    for path in nd2_parent_paths:
        path_str = str(path)
        if pattern.search(path_str):
            # Replace the zID in input folder name with the user's
            # zID (and append OME-TIFF)
            input_user = pattern.search(path_str).group()  # type: ignore
            out_path = path_str.replace(input_user, f"/{user}")
            out_path = Path(out_path) / "OME-TIFF"  # type: ignore
            logger.info(f"zID in input path does not match user's zID, adjusting output path to {str(out_path)}")
            out_paths.append(out_path)
        else:
            # Or if zIDs match, just add the original path
            # to the output path list (appending OME-TIFF)
            logger.debug(f"zID in input path matches user's zID, output path is {str(out_path)}")
            out_paths.append(path / "OME-TIFF")  # type: ignore

    # read template from file
    if template_path is None:
        jobscript_template = read_template("convert_nd2_pbs.sh")
    else:
        jobscript_template = Path(template_path).read_text()

    # create jobscripts using template
    for im_par_path, out_path, job_path in zip(nd2_parent_paths, out_paths, job_paths):
        jobscript = generate_pbs_script(
            template=jobscript_template,
            input_dir=str(im_par_path.resolve()),
            output_dir=str(out_path.resolve()),  # type: ignore
            log_dir=str(log_path.resolve()),
            user=user,
            email=email,
            n_batches=int(n_batches),
            mip=mip,
            keep_stacks=keep_stacks,
            y_direction=y_direction,
            channel_names=channel_names,
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
