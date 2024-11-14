from typing import Union, Optional
from pathlib import Path
import os
import shutil
import logging
import tempfile

from tqdm import tqdm
import requests

from blimp.constants import SCRIPTS_DIR

Path_t = Union[str, Path]

logger = logging.getLogger(__name__)


def load_example_data(data_dir: Optional[Path_t] = None) -> Path_t:
    """
    Download example data to ``data_dir``.

    Parameters
    ----------
    data_dir
        Defaults to ``notebooks/_data``.

    Returns
    -------
        Path to folder where dataset is stored.
    """
    from pathlib import Path

    fname = "_data"
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "notebooks"

    folder_dir = load_dataset(
        dataset_path=data_dir,
        fname=fname,
        backup_url="https://figshare.com/ndownloader/files/38934461",
    )

    return folder_dir


def download_figshare_file(repository_id: str, output_filename: str) -> str:
    """
    Download a file from Figshare using the repository ID.

    Parameters
    ----------
    repository_id : str
        The repository ID for the file on Figshare.
    output_filename : str
        The name of the output file.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    logger.info(f"Starting download from Figshare repository ID: {repository_id}")

    # Step 1: Get metadata of the repository
    metadata_url = f"https://api.figshare.com/v2/articles/{repository_id}"
    logger.debug(f"Fetching metadata from URL: {metadata_url}")
    response = requests.get(metadata_url)
    response.raise_for_status()  # Raise an error for bad requests
    metadata = response.json()
    logger.info("Metadata fetched successfully")

    # Step 2: Get the download URL of the first file
    files = metadata.get("files", [])
    if not files:
        logger.error("No files found in the repository.")
        raise Exception("No files found in the repository.")

    download_url = files[0].get("download_url")
    if not download_url:
        logger.error("No download URL found for the file.")
        raise Exception("No download URL found for the file.")

    logger.debug(f"Download URL: {download_url}")

    # Step 3: Download the file
    logger.info(f"Starting file download from: {download_url}")
    file_response = requests.get(download_url, stream=True)
    file_response.raise_for_status()
    logger.info("File downloaded successfully")

    # Step 4: Save the file
    file_name = files[0].get("name", "")
    file_extension = Path(file_name).suffix
    output_path = Path(output_filename)
    if output_path.suffix != file_extension:
        output_path = output_path.with_suffix(file_extension)

    CHUNK_SIZE = 65536  # 64KB
    BUFFER_SIZE = 1048576  # 1MB
    logger.info(f"Saving file to: {output_path}")
    with output_path.open("wb", buffering=BUFFER_SIZE) as f:
        for chunk in file_response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)

    logger.info(f"File saved successfully as {output_path}")
    return str(output_path)


def load_test_data():
    """
    Download test data to ``SCRIPTS_DIR/tests``.
    """
    repository_id = "23972244"
    base_dir = os.path.join(SCRIPTS_DIR, "tests")
    print(f"base_dir = {base_dir}")
    archive_path = os.path.join(base_dir, "_data.zip")
    # check if is downloaded already
    if os.path.exists(os.path.join(base_dir, "_data")) and os.path.exists(os.path.join(base_dir, "_experiments")):
        datacontent = os.listdir(os.path.join(base_dir, "_data", "resources"))
        if "affine.txt" in datacontent:
            logger.info(
                f"Using previously downloaded/extracted test data. Delete {os.path.join(base_dir, '_data')} directory to force re-download."
            )
            return
    # have to unpack/redownload
    if os.path.exists(archive_path):
        logger.info(f"Extracting test data. Delete {archive_path} directory to force re-download.")
        shutil.unpack_archive(archive_path, base_dir)
    else:
        logger.info(f"{archive_path} or dataset does not yet exist. Attempting to download from Figshare...")
        archive_path = download_figshare_file(repository_id, output_filename=archive_path)
        shutil.unpack_archive(archive_path, base_dir)
    return


def load_dataset(dataset_path: Path_t, fname: str, backup_url: str) -> Path_t:
    """
    Load dataset (from URL).

    In dataset_path, creates hierarchy of folders "raw", "archive".
    If unpacked files are already stored in "raw" doesn't do anything.
    Otherwise checks for archive file in "archive" folder and unpacks it into "raw" folder.
    If no files are present there, attempts to load the dataset from URL
    into "archive" folder and then unpacks it into "raw" folder.

    Parameters
    ----------
    dataset_path
        Path where folder for the dataset will be created.
    fname
        Desired name of the dataset
    backup_url
        Link from which dataset will be loaded

    Returns
    -------
    path to a folder where unpacked dataset is stored
    """
    unpacked_dir = Path(os.path.join(dataset_path, fname, "raw"))
    archive_path = Path(os.path.join(dataset_path, fname, "archive", f"{fname}.zip"))

    print(f"unpacked_dir = {unpacked_dir}")
    print(f"archive_path = {archive_path}")

    os.makedirs(unpacked_dir, exist_ok=True)
    foldercontent = os.listdir(str(unpacked_dir))
    if "operetta_cls_multiplex" in foldercontent:
        return unpacked_dir

    elif archive_path.exists():
        shutil.unpack_archive(archive_path, unpacked_dir)
        return unpacked_dir

    elif not archive_path.exists():
        if backup_url is None:
            raise Exception(
                f"File or directory {archive_path} does not exist and no backup_url was provided.\n"
                f"Please provide a backup_url or check whether path is spelled correctly."
            )

        print("Path or dataset does not yet exist. Attempting to download...")

        download(
            backup_url,
            output_path=archive_path,
        )

        shutil.unpack_archive(archive_path, unpacked_dir)

    return unpacked_dir


def getFilename_fromCd(cd):
    """
    Get filename from content-disposition or url request.
    """
    import re

    if not cd:
        return None
    fname = re.findall("filename=(.+)", cd)
    if len(fname) == 0:
        return None
    fname = fname[0]
    if '"' in fname:
        fname = fname.replace('"', "")
    return fname


def download(
    url: str,
    output_path: Optional[Path_t] = None,
    block_size: int = 1024,
    overwrite: bool = False,
) -> None:
    """
    Download a dataset irrespective of the format.

    Parameters
    ----------
    url
        URL to download
    output_path
        Path to download/extract the files to
    block_size
        Block size for downloads in bytes (default: 1024)
    overwrite
        Whether to overwrite existing files (default: False)
    """
    if output_path is None:
        output_path = tempfile.gettempdir()

    response = requests.get(url, stream=True)
    filename = getFilename_fromCd(response.headers.get("content-disposition"))

    # currently supports zip, tar, gztar, bztar, xztar
    download_to_folder = Path(output_path).parent
    os.makedirs(download_to_folder, exist_ok=True)

    archive_formats, _ = zip(*shutil.get_archive_formats())
    is_archived = str(Path(filename).suffix)[1:] in archive_formats
    assert is_archived

    download_to_path = os.path.join(download_to_folder, filename)

    if Path(download_to_path).exists():
        warning = f"File {download_to_path} already exists!"
        if not overwrite:
            print(warning)
            return
        else:
            print(f"{warning} Overwriting...")

    total = int(response.headers.get("content-length", 0))

    print(f"Downloading... {total}")
    with open(download_to_path, "wb") as file:
        for data in tqdm(response.iter_content(block_size)):
            file.write(data)

    os.replace(download_to_path, str(output_path))
