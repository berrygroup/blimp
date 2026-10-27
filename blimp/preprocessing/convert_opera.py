"""Convert Perkin-Elmer (Revvity) Opera Phenix image files to standard open
microscopy environment formats."""
from typing import List, Union, Optional
from pathlib import Path
import re
import glob
import logging
import subprocess

from blimp.utils import read_template, user_output_path, pbs_array_directive
from blimp.preprocessing.opera_to_ome_tiff import opera_regex_filename_pattern
from blimp.preprocessing.harmony_to_ome_ngff import (
    check_harmony_ngff,
    generate_pbs_script_ngff,
    ensure_harmony_plate_exists,
)

logger = logging.getLogger(__name__)

# converted OME-TIFFs no longer carry plane/channel/time information
ome_tiff_regex_filename_pattern = re.compile(r"r(?P<Row>\d+)c(?P<Col>\d+)f(?P<FieldID>\d+)(-mip)?\.ome\.tiff$")


def find_images_dirs(basepath: Union[str, Path]):
    """Recursively searches for opera "images" directories, identified by
    the "index.xml" metadata file they contain.

    Parameters
    ----------
    basepath
        root directory to begin searching

    Returns
    -------
    list of full paths to "images" directories
    """
    basepath = str(basepath)
    return sorted(str(Path(p).parent) for p in glob.glob(basepath + "/**/images/index.xml", recursive=True))


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
    save_metadata_files: bool,
    conda_env: str = "berrylab-py311",
) -> str:
    """Formats a PBS jobscript template using input arguments.

    Parameters
    ----------
    template
        PBS jobscript template
    input_dir
        full path to the measurement directory (containing "images")
    output_dir
        full path to the output OME-TIFF directory
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
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    conda_env
        name of the conda environment to activate on the compute node

    Returns
    -------
    Template as a formatted string to be written to file
    """
    array_directive, batch_id_expr = pbs_array_directive(n_batches)

    return template.format(
        INPUT_DIR=input_dir,
        OUTPUT_DIR=output_dir,
        LOG_DIR=log_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        ARRAY_DIRECTIVE=array_directive,
        BATCH_ID_EXPR=batch_id_expr,
        MIP="--mip" if mip else "",
        KEEP_STACKS="--keep_stacks" if keep_stacks else "",
        SAVE_METADATA_FILES="--save_metadata_files" if save_metadata_files else "",
        CONDA_ENV=conda_env,
    )


def convert_opera(
    in_path: Union[str, Path],
    job_path: Union[str, Path],
    image_format: str,
    template_path: Union[str, Path, None] = None,
    n_batches: int = 1,
    save_metadata_files: bool = True,
    mip: bool = False,
    keep_stacks: bool = True,
    y_direction: Optional[str] = None,
    x_direction: Optional[str] = None,
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    submit: bool = False,
    user: str = "z1234567",
    email: str = "foo@bar.com",
    conda_env: str = "berrylab-py311",
    dryrun: bool = False,
) -> None:
    """Recursively searches for opera 'images' directories and creates a job
    script to convert opera-generated TIFFs to OME-TIFF (via
    ``opera_to_ome_tiff``) or stitch them into a whole-plate OME-NGFF store
    (via ``harmony_to_ome_ngff``). Optionally submits the jobs.

    Parameters
    ----------
    in_path
        path to search for "images" directories
    job_path
        path where the jobscripts should be saved
    image_format
        "TIFF" or "NGFF"
    template_path
        path to a template for the PBS jobscript
        (default templates/convert_opera_pbs.sh for "TIFF",
        templates/convert_opera_ngff_pbs.sh for "NGFF")
    n_batches
        number of batches into which the processing should be split
        (by imaging site for "TIFF", by well for "NGFF").
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    mip
        whether to save maximum-intensity-projections
    keep_stacks
        whether to save stacks
    y_direction, x_direction, placement, channel_names
        "NGFF" only -- see
        :func:`blimp.preprocessing.harmony_to_ome_ngff.get_field_layout_from_harmony_metadata`.
    submit
        whether to also submit the batch jobs to the cluster
    user
        username (your zID, must match the path to your data)
    email
        email address for job notifications
    conda_env
        name of the conda environment to activate on the compute node
    dryrun
        prepare scripts and echo commands without submitting
    """

    if image_format not in {"TIFF", "NGFF"}:
        raise NotImplementedError(f'image_format = "{image_format}", only "TIFF" or "NGFF" are supported')

    # create job/log directory if it does not exist
    job_path = Path(job_path)
    log_path = job_path / "log"
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Jobscripts will be written to {job_path.resolve()}")

    # search recursively for "images" directories
    images_paths = find_images_dirs(in_path)
    images_parent_paths = [Path(p).parent for p in images_paths]
    jobscript_paths = [job_path / ("batch_convert_opera_" + str(p.name) + ".pbs") for p in images_parent_paths]

    # read template from file
    if template_path is None:
        template_name = "convert_opera_pbs.sh" if image_format == "TIFF" else "convert_opera_ngff_pbs.sh"
        jobscript_template = read_template(template_name)
    else:
        jobscript_template = Path(template_path).read_text()

    # create jobscripts using template
    for images_path, images_parent_path, jobscript_path in zip(images_paths, images_parent_paths, jobscript_paths):
        if image_format == "TIFF":
            jobscript = generate_pbs_script(
                template=jobscript_template,
                input_dir=str(images_parent_path.resolve()),
                output_dir=str(user_output_path(images_parent_path.resolve(), user, "OME-TIFF")),
                log_dir=str(log_path.resolve()),
                user=user,
                email=email,
                n_batches=int(n_batches),
                mip=mip,
                keep_stacks=keep_stacks,
                save_metadata_files=save_metadata_files,
                conda_env=conda_env,
            )
        else:
            plate_path = user_output_path(images_parent_path.resolve(), user, "plate.zarr")
            ensure_harmony_plate_exists(images_path, plate_path, instrument="opera")
            jobscript = generate_pbs_script_ngff(
                template=jobscript_template,
                input_dir=str(images_parent_path.resolve()),
                plate_path=str(plate_path),
                log_dir=str(log_path.resolve()),
                user=user,
                email=email,
                n_batches=int(n_batches),
                mip=mip,
                keep_stacks=keep_stacks,
                save_metadata_files=save_metadata_files,
                y_direction=y_direction,
                x_direction=x_direction,
                placement=placement,
                channel_names=channel_names,
                conda_env=conda_env,
            )
        # write to files
        with open(jobscript_path, "w+") as f:
            f.writelines(jobscript)

    # dryrun
    if dryrun:
        for j in jobscript_paths:
            logger.info(f"[dryrun] qsub {j}")

    # submit jobs
    if submit:
        for j in jobscript_paths:
            subprocess.run(["qsub", str(j)], check=True)

    return None


def check_convert_opera(
    in_path: Union[str, Path],
    mip: bool = False,
    keep_stacks: bool = True,
    save_metadata_files: bool = True,
    user: Optional[str] = None,
    image_format: str = "TIFF",
) -> None:
    """Check that all files were converted.

    Parameters
    ----------
    in_path
        path to search for "images" directories
    mip
        whether to also check for maximum-intensity-projections
    keep_stacks
        whether to also check for stacks
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    user
        zID of the user who ran the conversion. Output is looked for under
        this user's directory (see :func:`blimp.utils.user_output_path`).
    image_format
        "TIFF" or "NGFF" -- which conversion output to check
    """
    # search recursively for "images" directories
    logger.info("Searching for 'images' directories...")
    images_paths = find_images_dirs(in_path)
    logger.info(f"Found {len(images_paths)} 'images' directories...")

    for images_dir in images_paths:
        if image_format == "NGFF":
            check_harmony_ngff(
                in_path=images_dir,
                plate_path=user_output_path(Path(images_dir).parent.resolve(), user, "plate.zarr"),
                instrument="opera",
                mip=mip,
                keep_stacks=keep_stacks,
            )
            continue
        check_convert_opera_dir(
            in_path=images_dir,
            mip=mip,
            keep_stacks=keep_stacks,
            save_metadata_files=save_metadata_files,
            user=user,
        )

    return


def _get_sites(paths) -> set:
    """Get the set of unique (row, column, field) imaging sites from a list
    of opera input or OME-TIFF output filenames."""
    sites = set()
    for p in paths:
        m = opera_regex_filename_pattern.match(Path(p).name) or ome_tiff_regex_filename_pattern.match(Path(p).name)
        if m is not None:
            sites.add((int(m.group("Row")), int(m.group("Col")), int(m.group("FieldID"))))
    return sites


def _format_site(site: tuple) -> str:
    return (
        f"Plate row {site[0]}; plate col {site[1]}; field {site[2]} "
        + f"(filename string = r{str(site[0]).zfill(2)}c{str(site[1]).zfill(2)}f{str(site[2]).zfill(2)})"
    )


def check_convert_opera_dir(
    in_path: Union[str, Path],
    mip: bool = False,
    keep_stacks: bool = True,
    save_metadata_files: bool = True,
    user: Optional[str] = None,
) -> set:
    """Check that all files were converted (single directory).

    Parameters
    ----------
    in_path
        path to the "images" directory (including "images")
    mip
        whether to also check for maximum-intensity-projections
    keep_stacks
        whether to also check for stacks
    save_metadata_files
        whether the metadata files should be saved after XML parsing
    user
        zID of the user who ran the conversion. Output is looked for under
        this user's directory (see :func:`blimp.utils.user_output_path`).

    Returns
    -------
    Set of missing (non-converted) imaging sites as (row, column, field)
    """

    in_path = Path(in_path)
    out_path = user_output_path(in_path.parent, user, "OME-TIFF")
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
        for metadata_path in [out_path, out_path_mip] if mip else [out_path]:
            metadata_csv = metadata_path / "image_metadata.csv"
            metadata_pkl = metadata_path / "image_metadata.pkl"
            if (not metadata_csv.exists()) or (not metadata_pkl.exists()):
                logger.warning(f"Metadata files not found in {str(metadata_path)}")
            else:
                logger.info(f"Metadata files found in {str(metadata_path)}")

    # get a list of imaging sites from filenames in the well subdirectories of in_path
    unique_input_sites = _get_sites(in_path.glob("*/*.tiff"))
    logger.info(f"Images corresponding to {len(unique_input_sites)} unique sites found in {in_path}")

    missing_sites: set = set()
    output_paths = ([out_path] if keep_stacks else []) + ([out_path_mip] if mip else [])
    if len(output_paths) == 0:
        logger.warning("Neither mip nor keep_stacks are true. No images will be checked.")
    for output_path in output_paths:
        logger.info(f"Checking output directory {str(output_path)}...")
        unique_output_sites = _get_sites(output_path.glob("*.ome.tiff"))
        missing_sites_output = unique_input_sites.difference(unique_output_sites)
        if len(missing_sites_output) != 0:
            logger.warning(f"The following sites are expected in {str(output_path)}, but were not found")
            for m in sorted(missing_sites_output):
                logger.warning(_format_site(m))
        else:
            logger.info(
                f"Images corresponding to {len(unique_output_sites)} unique imaging sites found in {output_path}"
            )
            logger.info("Conversion appears complete")
        missing_sites = missing_sites.union(missing_sites_output)

    return missing_sites
