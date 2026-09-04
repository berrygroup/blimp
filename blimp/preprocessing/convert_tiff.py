"""Generate a PBS job to assemble an existing OME-TIFF pipeline's output
(one plate's worth of wells) into a whole-plate OME-NGFF store, via
``tiff_to_ome_ngff.py``'s own CLI.

Unlike ``convert_nd2.py``/``convert_operetta.py``, this has no recursive
multi-plate discovery: by the time a TIFF pipeline exists, the caller
already knows exactly which plate's output they want assembled, so
``in_path``/``plate_path`` are explicit, and the command runs once per
plate -- matching ``tiff_to_ome_ngff()``'s own shape (it already takes an
explicit ``in_path``/``plate_path`` and internally batches over however
many wells' ``*_metadata.csv`` files live under that one ``in_path``).
"""
from typing import List, Union, Optional
from pathlib import Path
import logging
import subprocess

from blimp.utils import read_template
from blimp.ome_ngff import ensure_plate_exists
from blimp.preprocessing.tiff_to_ome_ngff import _discover_well_manifest

logger = logging.getLogger(__name__)


def generate_pbs_script_tiff_ngff(
    template: str,
    input_dir: str,
    plate_path: str,
    log_dir: str,
    user: str,
    email: str,
    n_batches: int,
    y_direction: str,
    x_direction: str,
    placement: str,
    channel_names: Union[str, List[str], None] = None,
    label_dir: Optional[str] = None,
    feature_csv_dir: Optional[str] = None,
    point_object_channel_names: Optional[List[str]] = None,
    conda_env: str = "berrylab-py311",
) -> str:
    """Formats a PBS jobscript template using
    ``tiff_to_ome_ngff.py``'s own CLI -- mirrors
    :func:`blimp.preprocessing.convert_nd2.generate_pbs_script_ngff`.

    Parameters
    ----------
    template
        PBS jobscript template
    input_dir
        full path to the directory of field TIFFs + metadata CSVs
    plate_path
        full path to the shared plate .zarr store (created up front by
        :func:`convert_tiff` before this jobscript is written)
    log_dir
        full path to where output logs should be written
    user
        usename (zID on katana) for job submission and location of scripts
    email
        email address for notifications
    n_batches
        how many batches (by well) into which processing should be split
    y_direction, x_direction, placement
        see :func:`blimp.preprocessing.tiff_to_ome_ngff.get_field_layout_from_tiff_metadata`
    channel_names
        List of channel names in case those found in the TIFF metadata are
        incorrect
    label_dir, feature_csv_dir, point_object_channel_names
        see :func:`blimp.preprocessing.tiff_to_ome_ngff.convert_tiff_well_to_ome_ngff`
    conda_env
        name of the conda environment to activate on the compute node

    Returns
    -------
    Template as a formatted string to be written to file
    """
    if channel_names is None:
        channel_names_str = ""
    else:
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        channel_names_str = "--channel_names " + (" ".join(channel_names))

    if point_object_channel_names is None:
        point_object_channel_names_str = ""
    else:
        point_object_channel_names_str = "--point_object_channel_names " + (" ".join(point_object_channel_names))

    return template.format(
        INPUT_DIR=input_dir,
        PLATE_PATH=plate_path,
        LOG_DIR=log_dir,
        USER=user,
        USER_EMAIL=email,
        N_BATCHES=n_batches,
        BATCH_MAX=n_batches - 1,
        Y_DIRECTION=y_direction,
        X_DIRECTION=x_direction,
        PLACEMENT=placement,
        CHANNEL_NAMES=channel_names_str,
        LABEL_DIR=f"--label_dir {label_dir}" if label_dir else "",
        FEATURE_CSV_DIR=f"--feature_csv_dir {feature_csv_dir}" if feature_csv_dir else "",
        POINT_OBJECT_CHANNEL_NAMES=point_object_channel_names_str,
        CONDA_ENV=conda_env,
    )


def convert_tiff(
    in_path: Union[str, Path],
    plate_path: Union[str, Path],
    image_format: str = "NGFF",
    plate_name: Optional[str] = None,
    label_dir: Union[str, Path, None] = None,
    feature_csv_dir: Union[str, Path, None] = None,
    point_object_channel_names: Optional[List[str]] = None,
    template_path: Union[str, Path, None] = None,
    n_batches: int = 1,
    y_direction: str = "down",
    x_direction: str = "left",
    placement: str = "grid",
    channel_names: Union[str, List[str], None] = None,
    job_path: Union[str, Path] = ".",
    submit: bool = False,
    user: str = "z1234567",
    email: str = "foo@bar.com",
    conda_env: str = "berrylab-py311",
    dryrun: bool = False,
) -> None:
    """Creates a PBS job script to assemble one plate's existing OME-TIFF
    pipeline output into a whole-plate OME-NGFF store. Optionally submits
    the job.

    Validates ``label_dir``/``feature_csv_dir`` up front, before writing
    anything: an HPC job can queue for hours before it actually runs, so a
    typo'd path should fail immediately here, not silently produce an empty
    result (or wait in a PBS queue) before anyone notices. A path that
    exists and has at least one matching file across every well/field found
    under ``in_path`` is fine even if it's missing some -- that's a normal
    partial-coverage case, and ``_discover_well_manifest`` already logs a
    warning naming exactly which fields are missing from which well.

    Parameters
    ----------
    in_path
        Directory containing the intensity field TIFFs and metadata
        CSVs for one plate (e.g. an ``OME-TIFF-MIP/`` folder) -- may
        hold many wells, one ``*_metadata.csv`` file each.
    plate_path
        Full path to the shared plate .zarr store to create/write to.
    image_format
        Must be ``"NGFF"`` -- kept as a parameter for consistency with
        ``convert_nd2``/``convert_operetta``'s own ``image_format``, even
        though there is currently only one sensible value for this path.
    plate_name
        Name for the plate, used only if it does not already exist
        (default: derived from ``plate_path``'s own stem).
    label_dir
        Directory containing one (possibly multi-channel) label TIFF per
        field. Omit to skip labels entirely.
    feature_csv_dir
        Directory containing one already-aggregated ``quantify()``
        measurement CSV per field. Omit to skip feature tables entirely.
    point_object_channel_names
        See :func:`blimp.preprocessing.tiff_to_ome_ngff.convert_tiff_well_to_ome_ngff`.
    template_path
        path to a template for the PBS jobscript (default
        ``templates/convert_tiff_ngff_pbs.sh``)
    n_batches
        number of batches (by well) into which the processing should be split
    y_direction, x_direction, placement
        See :func:`blimp.preprocessing.tiff_to_ome_ngff.get_field_layout_from_tiff_metadata`.
    channel_names
        List of channel names in case those found in the TIFF metadata are
        incorrect
    job_path
        path where the jobscript should be saved (logs are saved in the
        ``log`` subdirectory of this path)
    submit
        whether to also submit the job to the cluster
    user
        username (your zID)
    email
        email address for job notifications
    conda_env
        name of the conda environment to activate on the compute node
    dryrun
        prepare the script and echo the command without submitting
    """
    if image_format != "NGFF":
        raise NotImplementedError(f'image_format = "{image_format}", only "NGFF" is supported')

    in_path = Path(in_path)
    plate_path = Path(plate_path)
    job_path = Path(job_path)
    log_path = job_path / "log"
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Jobscript will be written to {job_path.resolve()}")

    nd2_stems = sorted(p.stem[: -len("_metadata")] for p in in_path.glob("*_metadata.csv"))
    if not nd2_stems:
        raise FileNotFoundError(f"No '*_metadata.csv' files found under {in_path}")
    logger.info(f"Found {len(nd2_stems)} well(s) under {in_path}")

    if label_dir is None:
        logger.warning("No label_dir given: the resulting store will have no label layers.")
    if feature_csv_dir is None:
        logger.warning("No feature_csv_dir given: labels will have no attached feature values.")

    label_found = label_dir is None
    feature_found = feature_csv_dir is None
    for nd2_stem in nd2_stems:
        manifest = _discover_well_manifest(nd2_stem, in_path, label_dir, feature_csv_dir)
        if label_dir is not None and manifest["label_exists"].any():
            label_found = True
        if feature_csv_dir is not None and manifest["feature_exists"].any():
            feature_found = True
    if not label_found:
        raise FileNotFoundError(
            f"label_dir {label_dir} has no files matching any field across "
            f"{len(nd2_stems)} well(s) found under {in_path}"
        )
    if not feature_found:
        raise FileNotFoundError(
            f"feature_csv_dir {feature_csv_dir} has no files matching any field across "
            f"{len(nd2_stems)} well(s) found under {in_path}"
        )

    # Created up front (idempotent, and cheap -- see ensure_plate_exists) so
    # the parallel batch tasks this jobscript's #PBS -J array launches never
    # race to create it.
    ensure_plate_exists(plate_path, plate_name or plate_path.stem)

    if template_path is None:
        jobscript_template = read_template("convert_tiff_ngff_pbs.sh")
    else:
        jobscript_template = Path(template_path).read_text()

    jobscript = generate_pbs_script_tiff_ngff(
        template=jobscript_template,
        input_dir=str(in_path.resolve()),
        plate_path=str(plate_path.resolve()),
        log_dir=str(log_path.resolve()),
        user=user,
        email=email,
        n_batches=int(n_batches),
        y_direction=y_direction,
        x_direction=x_direction,
        placement=placement,
        channel_names=channel_names,
        label_dir=str(Path(label_dir).resolve()) if label_dir is not None else None,
        feature_csv_dir=str(Path(feature_csv_dir).resolve()) if feature_csv_dir is not None else None,
        point_object_channel_names=point_object_channel_names,
        conda_env=conda_env,
    )

    job_script_path = job_path / f"batch_convert_tiff_{in_path.stem}.pbs"
    with open(job_script_path, "w+") as f:
        f.writelines(jobscript)

    if dryrun:
        logger.info(f"[dryrun] qsub {job_script_path}")

    if submit:
        subprocess.run(["qsub", str(job_script_path)], check=True)

    return None
