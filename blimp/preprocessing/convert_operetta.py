"""Convert Perkin-Elmer Operetta image files to standard open microscopy
environment formats."""
from typing import Union
from pathlib import Path
import os
import re
import glob
import logging

from blimp.utils import read_template

logger = logging.getLogger(__name__)


def find_images_dirs(basepath: Union[str, Path]):
    basepath = str(basepath)
    return glob.glob(basepath + "/**/Images", recursive=True)


def generate_pbs_script(
    template: str,
    input_dir: str,
    log_dir: str,
    user: str,
    email: str,
    n_batches: int,
    mip: bool,
    keep_stacks: bool,
    save_metadata_files: bool,
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
    keep_stacks
        whether to save stacks

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
        MIP="--mip" if mip else "",
        KEEP_STACKS="--keep_stacks" if keep_stacks else "",
        SAVE_METADATA_FILES="--save_metadata_files" if save_metadata_files else "",
    )


def convert_operetta(
    in_path: Union[str, Path],
    job_path: Union[str, Path],
    image_format: str,
    template_path: Union[str, Path, None] = None,
    n_batches: int = 1,
    save_metadata_files: bool = True,
    mip: bool = False,
    keep_stacks: bool = True,
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
        path to search for "Images" directories
    out_path
        path where the converted image data should be saved
    job_path
        path where the jobscripts should be saved
    metadata_file
        name of the xml metadata file inside "Images"
    image_format
        "TIFF" or "NGFF" (currently only TIFF implemented)
    template_path
        path to a template for the PBS jobscript
        (default templates/convert_operetta_pbs.sh)
    n_batches
        number of batches into which the processing should be split.
    batch_id
        current batch to process
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    submit
        whether to also submit the batch jobs to the cluster
    user
        username (your zID, must match the path to your data)
    email
        email address for job notifications
    dryrun
        prepare scripts and echo commands without submitting
    """

    # create job/log directory if it does not exist
    job_path = Path(job_path)
    log_path = job_path / "log"
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Jobscripts will be written to {job_path.resolve()}")

    # search recursively for "Images" directories
    images_paths = find_images_dirs(in_path)
    images_parent_paths = [Path(p).parent for p in images_paths]
    jobscript_paths = [job_path / ("batch_convert_operetta_" + str(p.stem) + ".pbs") for p in images_parent_paths]

    # read template from file
    if template_path is None:
        jobscript_template = read_template("convert_operetta_pbs.sh")
    else:
        jobscript_template = Path(template_path).read_text()

    # create jobscripts using template
    for images_parent_path, jobscript_path in zip(images_parent_paths, jobscript_paths):
        jobscript = generate_pbs_script(
            template=jobscript_template,
            input_dir=str(images_parent_path.resolve()),
            log_dir=str(log_path.resolve()),
            user=user,
            email=email,
            n_batches=int(n_batches),
            mip=mip,
            keep_stacks=keep_stacks,
            save_metadata_files=save_metadata_files,
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

    return None


def check_convert_operetta(
    in_path: Union[str, Path], mip: bool = False, keep_stacks: bool = True, save_metadata_files: bool = True
) -> None:
    """Check that all files were converted.

    Parameters
    ----------
    in_path
        path to search for "Images" directories
    mip
        whether to also check for maximum-intensity-projections
    keep_stacks
        whether to also check for stacks
    save_metadata_files
        whether the metadata files should be saved after XML parsing

    Returns
    -------
    List of missing (non-converted) files
    """
    # search recursively for "Images" directories
    logger.info("Searching for 'Images' directories...")
    images_paths = find_images_dirs(in_path)
    logger.info(f"Found {len(images_paths)} 'Images' directories...")

    for images_dir in images_paths:
        check_convert_operetta_dir(
            in_path=images_dir,
            mip=mip,
            keep_stacks=keep_stacks,
            save_metadata_files=save_metadata_files,
        )

    return


def check_convert_operetta_dir(
    in_path: Union[str, Path], mip: bool = False, keep_stacks: bool = True, save_metadata_files: bool = True
) -> set:
    """Check that all files were converted (single directory).

    Parameters
    ----------
    in_path
        path to the "Images" directory (including "Images")
    mip
        whether to also check for maximum-intensity-projections
    keep_stacks
        whether to also check for stacks
    save_metadata_files
        whether the metadata files should be saved after XML parsing

    Returns
    -------
    List of missing (non-converted) files
    """

    in_path = Path(in_path)
    out_path = in_path.parent / "OME-TIFF"
    out_path_mip = out_path.parent / (str(out_path.stem) + "-MIP")

    # check directories exist
    if not in_path.exists():
        raise FileNotFoundError
    if not out_path.exists():
        raise FileNotFoundError
    if mip and not out_path_mip.exists():
        raise FileNotFoundError

    # check metadata files written
    if save_metadata_files:
        metadata_csv = out_path / "image_metadata.csv"
        metadata_pkl = out_path / "image_metadata.pkl"
        if (not metadata_csv.exists()) or (not metadata_pkl.exists()):
            logger.warning(f"Metadata files not found in {str(out_path)}")
        else:
            logger.info(f"Metadata files found in {str(out_path)}")
        if mip:
            metadata_csv = out_path_mip / "image_metadata.csv"
            metadata_pkl = out_path_mip / "image_metadata.pkl"
            if (not metadata_csv.exists()) or (not metadata_pkl.exists()):
                logger.warning(f"Metadata files not found in {str(out_path_mip)}")
            else:
                logger.info(f"Metadata files found in {str(out_path_mip)}")

    # get a list of imaging sites from filenames in in_path
    unique_input_sites = {tuple(re.findall(r"(?<=r|c|f)\d+", str(s))) for s in in_path.glob("*.tiff")}
    unique_output_sites = {tuple(re.findall(r"(?<=r|c|f)\d+", str(s))) for s in out_path.glob("*.tiff")}

    missing_sites = unique_input_sites.difference(unique_output_sites)

    logger.info(f"Checking output directory...")
    if len(missing_sites) != 0:
        logger.warn(f"The following sites are expected in {str(out_path)}, but were not found")
        for m in missing_sites:
            logger.warn(
                f"Plate row {m[0]}; plate col {m[1]}; field {m[2]} (filename string = r{str(m[0]).zfill(2)}c{str(m[1]).zfill(2)}f{str(m[2]).zfill(2)})"
            )
    else:
        logger.info(f"Images corresponding to {len(unique_input_sites)} unique sites found in {in_path}")
        logger.info(f"Images corresponding to {len(unique_output_sites)} unique imaging sites found in {out_path}")
        logger.info(f"Conversion appears complete")

    if mip:
        logger.info(f"Checking MIP output directory...")
        unique_output_sites_mip = {tuple(re.findall(r"(?<=r|c|f)\d+", str(s))) for s in out_path_mip.glob("*.tiff")}
        missing_sites_mip = unique_input_sites.difference(unique_output_sites_mip)

        if len(missing_sites) != 0:
            logger.warn(f"The following sites are expected in {str(out_path_mip)}, but were not found")
            for m in missing_sites_mip:
                logger.warn(
                    f"Plate row {m[0]}; plate col {m[1]}; field {m[2]} (filename string = r{str(m[0]).zfill(2)}c{str(m[1]).zfill(2)}f{str(m[2]).zfill(2)})"
                )
        else:
            logger.info(f"Images corresponding to {len(unique_input_sites)} unique sites found in {in_path}")
            logger.info(
                f"Images corresponding to {len(unique_output_sites_mip)} unique imaging sites found in {out_path_mip}"
            )
            logger.info(f"Conversion appears complete")

    return missing_sites
