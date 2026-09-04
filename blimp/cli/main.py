import os
import argparse

from blimp.log import configure_logging
from blimp.cli.prepare_config import prepare_config

# fmt: off
header = """
    BLIMP

    This is a command-line interface (CLI) for the blimp
    python package (Berry Lab IMage Processing). blimp
    has a full API (github.com/berrygroup/blimp) for
    programmatic usage.
    """
# fmt: on


def _get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blimp", description=header, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (e.g. -vvv)",
    )
    return parser


def _get_setup_parser():
    parser = argparse.ArgumentParser(description="Create configuration file ``blimp.ini``.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="create default configuration file without asking for user input.",
    )
    return parser


def _add_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i", "--input_path", help="Top-level directory to search for images to be converted (required)", required=True
    )
    parser.add_argument(
        "-j",
        "--jobscript_path",
        default=os.getcwd(),
        help="Directory to save PBS jobscripts (default = current working directory)",
    )
    parser.add_argument(
        "--output_format",
        default="TIFF",
        help="Output format for images (TIFF or NGFF)",
    )
    parser.add_argument(
        "--mip",
        default=False,
        action="store_true",
        help="Whether to save maximum intensity projections? (default = False)",
    )
    parser.add_argument(
        "--keep_stacks",
        default=False,
        action="store_true",
        help="Whether to save image stacks (all z-planes)? (default = False)",
    )
    parser.add_argument("--user", metavar="ZID", help="Your zID for HPC submission (required)", required=True)
    parser.add_argument("--email", help="Email address for job notifications", required=False)
    parser.add_argument(
        "--batch",
        default=1,
        help="When using batch processing, provide the number of batches (default = 1)",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Whether to submit jobs after creating jobscripts (default = False)",
    )
    parser.add_argument(
        "--verify-only",
        default=False,
        action="store_true",
        help="""
        Check if the conversion is complete (default = False). Note that
        this option does not actually attempt to do any conversion, but
        instead returns information via logging statements as to whether
        all files have already been successfully converted. Use with
        '-vv' to show results.
    """,
    )
    return None


def _add_convert_operetta_args(parser: argparse.ArgumentParser) -> None:
    return None


def _add_convert_nd2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-y",
        "--y_direction",
        default="down",
        help="""
        Microscope stages can have inconsistent y orientations
        relative to images. Standardised field identifiers are
        derived from microscope stage positions but to ensure
        the orientation of the y-axis relative to images, this
        must be specified. Default value is "down" so that
        y-coordinate values increase as the stage moves toward
        the eyepiece. Change to "up" if stiching doesn't look
        right!
    """,
    )
    parser.add_argument(
        "-x",
        "--x_direction",
        default="left",
        help="""
        --output_format NGFF only: direction of increasing (stage)
        x-coordinates relative to the image ("left" or "right").
        Default "left" -- override per-instrument if a microscope
        needs "right" instead.
    """,
    )
    parser.add_argument(
        "--placement",
        default="grid",
        choices=["grid", "exact"],
        help="""
        --output_format NGFF only: "grid" (default) snaps fields to a
        tile grid, discarding stage jitter; "exact" uses the raw stage
        offset directly -- see nd2_to_ome_ngff.get_field_layout.
    """,
    )
    parser.add_argument(
        "-c",
        "--channel_names",
        type=str,
        nargs="+",
        default=None,
        help="List of channel names",
    )
    return None


def _add_convert_tiff_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input_path",
        help="Directory containing field TIFFs and metadata sidecars for one plate (required)",
        required=True,
    )
    parser.add_argument("-o", "--plate_path", help="Path to the shared plate .zarr store (required)", required=True)
    parser.add_argument("--plate_name", default=None, help="Name for the plate (default: derived from path)")
    parser.add_argument(
        "-j",
        "--jobscript_path",
        default=os.getcwd(),
        help="Directory to save PBS jobscripts (default = current working directory)",
    )
    parser.add_argument(
        "--output_format",
        default="NGFF",
        help="Output format (only NGFF is currently supported for this input type)",
    )
    parser.add_argument("--user", metavar="ZID", help="Your zID for HPC submission (required)", required=True)
    parser.add_argument("--email", help="Email address for job notifications", required=False)
    parser.add_argument(
        "--batch",
        default=1,
        help="When using batch processing, provide the number of batches, by well (default = 1)",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Whether to submit jobs after creating jobscripts (default = False)",
    )
    parser.add_argument(
        "-y",
        "--y_direction",
        default="down",
        help='Direction of increasing (stage) y-coordinates ("up" or "down")',
    )
    parser.add_argument(
        "-x",
        "--x_direction",
        default="left",
        help='Direction of increasing (stage) x-coordinates ("left" or "right")',
    )
    parser.add_argument(
        "--placement",
        default="grid",
        choices=["grid", "exact"],
        help='"grid" (default) snaps fields to a tile grid, discarding stage jitter; '
        '"exact" uses the raw stage offset directly',
    )
    parser.add_argument(
        "-c",
        "--channel_names",
        type=str,
        nargs="+",
        default=None,
        help="List of channel names",
    )
    parser.add_argument(
        "-l",
        "--label_dir",
        default=None,
        help="Directory containing one (possibly multi-channel) label TIFF per field",
    )
    parser.add_argument(
        "-f",
        "--feature_csv_dir",
        default=None,
        help="Directory containing one already-aggregated quantify() CSV per field",
    )
    parser.add_argument(
        "--point_object_channel_names",
        type=str,
        nargs="+",
        default=None,
        help="Label channel names, if any, with no stable per-pixel identity (see quantify()'s point_objects)",
    )
    return None


def _convert_tiff(args) -> None:
    """Wrapper for convert_tiff()"""
    from blimp.preprocessing.convert_tiff import convert_tiff

    convert_tiff(
        in_path=args.input_path,
        plate_path=args.plate_path,
        image_format=args.output_format,
        plate_name=args.plate_name,
        label_dir=args.label_dir,
        feature_csv_dir=args.feature_csv_dir,
        point_object_channel_names=args.point_object_channel_names,
        n_batches=args.batch,
        y_direction=args.y_direction,
        x_direction=args.x_direction,
        placement=args.placement,
        channel_names=args.channel_names,
        job_path=args.jobscript_path,
        submit=args.submit,
        user=args.user,
        email=args.email,
        dryrun=False,
    )
    return None


def _convert_nd2(args) -> None:
    """Wrapper for convert_nd2()"""
    from blimp.preprocessing.convert_nd2 import convert_nd2

    convert_nd2(
        in_path=args.input_path,
        job_path=args.jobscript_path,
        image_format=args.output_format,
        n_batches=args.batch,
        mip=args.mip,
        keep_stacks=args.keep_stacks,
        y_direction=args.y_direction,
        x_direction=args.x_direction,
        placement=args.placement,
        channel_names=args.channel_names,
        submit=args.submit,
        user=args.user,
        email=args.email,
        dryrun=False,
    )
    return None


def _convert_operetta(args) -> None:
    """Wrapper for convert_operetta()"""
    from blimp.preprocessing.convert_operetta import (
        convert_operetta,
        check_convert_operetta,
    )

    if args.verify_only:
        check_convert_operetta(
            in_path=args.input_path,
            save_metadata_files=True,
            mip=args.mip,
        )
    else:
        convert_operetta(
            in_path=args.input_path,
            job_path=args.jobscript_path,
            image_format=args.output_format,
            n_batches=args.batch,
            save_metadata_files=True,
            mip=args.mip,
            keep_stacks=args.keep_stacks,
            submit=args.submit,
            user=args.user,
            email=args.email,
            dryrun=False,
        )
    return None


def _add_archive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i", "--input_path", help="Top-level directory to search for images to be converted (required)", required=True
    )

    parser.add_argument(
        "-j",
        "--jobscript_path",
        default=None,
        help="Directory to save jobscripts (default = current working directory)",
    )

    parser.add_argument(
        "--first_name", help="Your first name for creating the archive path at UNSW_RDS (required)", required=True
    )

    parser.add_argument(
        "--project_name",
        default="D0419427",
        help="The project name where data should be archived on UNSW's Data Archive (default = D0419427)",
    )

    return None


def _archive_nd2(args) -> None:
    """Wrapper for convert_nd2(input_type=="nd2")"""
    from blimp.archive import archive

    archive(
        in_path=args.input_path,
        jobscript_path=args.jobscript_path if args.jobscript_path is not None else os.getcwd(),
        input_type="nd2",
        first_name=args.first_name,
        project_name=args.project_name,
    )
    return None


def _archive_operetta(args) -> None:
    """Wrapper for archive(input_type=="operetta")"""
    from blimp.archive import archive

    archive(
        in_path=args.input_path,
        jobscript_path=args.jobscript_path if args.jobscript_path is not None else os.getcwd(),
        input_type="operetta",
        first_name=args.first_name,
        project_name=args.project_name,
    )
    return None


def _get_full_parser() -> argparse.ArgumentParser:
    # ----------- #
    # BASE PARSER #
    # ----------- #

    parser = _get_base_parser()

    # --------------- #
    # LEVEL 1 PARSERS #
    # --------------- #

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    setup_header = """
    * setup: Create configuration file ``blimp.ini``.
    """
    setup_parser = subparsers.add_parser(
        "setup",
        help="Create configuration file ``blimp.ini``.",
        description="".join([header, setup_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_parser.add_argument(
        "--quiet",
        action="store_true",
        help="create default configuration file without asking for user input.",
    )
    setup_parser.set_defaults(func=prepare_config)

    convert_header = """
    * convert: Convert raw microscope files to standard
      image formats such as OME-TIFF and OME-NGFF.
    """
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw microscope files to standard image formats",
        description="".join([header, convert_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    archive_header = """
    * archive: archive original files on UNSW's Data
      Archive.
    """
    archive_parser = subparsers.add_parser(
        "archive",
        help="Archive original files on UNSW's Data Archive.",
        description="".join([header, archive_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --------------- #
    # LEVEL 2 PARSERS #
    # --------------- #

    # Convert

    convert_subparsers = convert_parser.add_subparsers(dest="input_type", help="Input type")
    convert_subparsers.required = True

    # Convert - ND2
    convert_nd2_header = """
        * nd2: Convert Nikon nd2 files.
    """
    convert_nd2_subparser = convert_subparsers.add_parser(
        "nd2",
        help="Nikon nd2 file",
        description="".join([header, convert_header, convert_nd2_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    convert_nd2_subparser.set_defaults(func=_convert_nd2)
    _add_convert_args(convert_nd2_subparser)
    _add_convert_nd2_args(convert_nd2_subparser)

    # Convert - Operetta
    convert_operetta_header = """
        * operetta: Convert Perkin-Elmer Operetta files.
    """
    convert_operetta_subparser = convert_subparsers.add_parser(
        "operetta",
        help="Perkin-Elmer Operetta",
        description="".join([header, convert_header, convert_operetta_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    convert_operetta_subparser.set_defaults(func=_convert_operetta)
    _add_convert_args(convert_operetta_subparser)
    _add_convert_operetta_args(convert_operetta_subparser)

    # Convert - TIFF pipeline
    convert_tiff_header = """
        * tiff: Assemble an existing OME-TIFF pipeline's output
          (intensity, labels, features) into a whole-plate OME-NGFF store.
    """
    convert_tiff_subparser = convert_subparsers.add_parser(
        "tiff",
        help="Existing OME-TIFF pipeline output",
        description="".join([header, convert_header, convert_tiff_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    convert_tiff_subparser.set_defaults(func=_convert_tiff)
    _add_convert_tiff_args(convert_tiff_subparser)

    # Archive

    archive_subparsers = archive_parser.add_subparsers(dest="input_type", help="Input type")
    archive_subparsers.required = True

    # Archive - ND2
    archive_nd2_header = """
        * nd2: archive Nikon nd2 files.
    """
    archive_nd2_subparser = archive_subparsers.add_parser(
        "nd2",
        help="Nikon nd2 file",
        description="".join([header, archive_header, archive_nd2_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    archive_nd2_subparser.set_defaults(func=_archive_nd2)
    _add_archive_args(archive_nd2_subparser)

    # Archive - Operetta
    archive_operetta_header = """
        * operetta: archive Perkin-Elmer Operetta files.
    """
    archive_operetta_subparser = archive_subparsers.add_parser(
        "operetta",
        help="Perkin-Elmer Operetta",
        description="".join([header, archive_header, archive_operetta_header]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    archive_operetta_subparser.set_defaults(func=_archive_operetta)
    _add_archive_args(archive_operetta_subparser)

    return parser


def main():
    """Main BLIMP CLI entry point"""

    parser = _get_full_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    # call function provided as default for the subparser
    if args.func == prepare_config:
        # setup requires user input so needs a slightly different interface
        prepare_config(quiet=args.quiet)
    else:
        args.func(args)

    return


if __name__ == "__main__":
    main()
