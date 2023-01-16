"""Convert Nikon nd2 files to standard open microscopy environment formats."""
from typing import List, Union
from pathlib import Path
import os, stat
import re
import logging

import numpy as np

from blimp.preprocessing import find_nd2_files, find_images_dirs

logger = logging.getLogger(__name__)


def check_config_file() -> bool:
    """Check that the config file exists in the correct location"""
    config_path = Path.home() / "config.cfg"
    if not config_path.exists():
        logger.error(
            f"Config file not found at {config_path}. Please use get-config-file from UNSW's data archive module"
        )
        os._exit(1)
    else:
        return True


def write_archiving_script(
    file_paths: Union[List[Path],List[str]], script_path: Union[Path, str], first_name: str, project_name: str = "D0419427"
) -> None:
    """
    Create a bash script that execute the 'upload.sh' command for each file path in `file_paths` list.
    The script will be saved in the current working directory with name 'upload_script.sh'

    Parameters
    ----------
    file_paths : list of str
        list of file paths to be uploaded

    Returns
    -------
    None
    """

    file_paths = [str(f) for f in file_paths]

    with open(Path(script_path), "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("## Archiving script generated by BLIMP's archive tool\n\n")
        f.write("## Inputs:\n")
        f.write(f"##     First name: {first_name}\n")
        f.write(f"##     Project:    {project_name}\n\n")
        f.write("module add unswdataarchive/2020-03-19\n\n")
        f.write("export CONFIG_FILE=${HOME}/config.cfg\n\n")
        for file_path in file_paths:
            # use a regex to strip the first part of the filename that
            # we do not need to include in the upload destination
            relative_path = re.sub(r"^\/srv\/scratch\/berrylab\/z\d{7}\/", "", file_path)
            f.write(f"java -Dmf.cfg=$CONFIG_FILE -cp /apps/unswdataarchive/2020-03-19/aterm.jar arc.mf.command.Execute import -verbose true -import-empty-folders true -namespace /UNSW_RDS/{project_name}/{first_name}/{relative_path} {file_path}\n")


def archive(
    in_path: Union[Path, str], jobscript_path: Union[Path, str], first_name: str, project_name: str, input_type: str
) -> None:
    """
    Archive image files or directories using the specified input type.

    Parameters
    ----------
    in_path
        The path to the input image files or directories.
    job_path
        The path to the job directory where the archiving script will be written.
    input_type
        The type of input, either "nd2" or "operetta".

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If input_type is not "nd2" or "operetta".
    """

    # check_config_file()

    in_path = Path(in_path)
    jobscript_path = Path(jobscript_path) / "archive_data_blimp.sh"

    if input_type == "nd2":
        # Archive nd2 files individually since
        # these tend to be few in number
        image_files = find_nd2_files(in_path)
        write_archiving_script(image_files, jobscript_path, first_name, project_name)

    elif input_type == "operetta":
        # Archive 'Images' directories together
        # with associated metadata folders
        image_dirs = find_images_dirs(in_path)
        sub_dirs = ["Images", "Assaylayout", "FFC_Profile"]
        archive_dirs = np.concatenate([[Path(d).parent / s for s in sub_dirs] for d in image_dirs])
        write_archiving_script(archive_dirs.flatten(), jobscript_path, first_name, project_name)

    else:
        logger.error(f"input_type {input_type} not recognised. Please specify 'nd2' or 'operetta' input_type.")
        raise ValueError("Input type not recognised")

    os.chmod(jobscript_path,stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)

    return
