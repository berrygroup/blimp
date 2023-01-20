import os
import sys
import argparse

from blimp.log import configure_logging

from .prepare_config import prepare_config

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
        help="Directory to save PBS jobscripts (default = current working directory",
        required=True,
    )
    parser.add_argument(
        "--output_format",
        default="TIFF",
        help="Output format for images (TIFF or NGFF, currently only TIFF implemented)",
    )
    parser.add_argument(
        "-m",
        "--mip",
        default=False,
        action="store_true",
        help="Whether to save maximum intensity projections? (default = False)",
    )
    parser.add_argument("--user", metavar="ZID", help="Your zID for HPC submission(required)", required=True)
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
        help="whether to submit jobs after creating jobscripts",
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
    return None


def _convert_nd2(args) -> None:
    """Wrapper for convert_nd2()"""
    from blimp.preprocessing import convert_nd2

    convert_nd2(
        in_path=args.input_path,
        job_path=args.jobscript_path,
        image_format=args.output_format,
        n_batches=args.batch,
        mip=args.mip,
        y_direction=args.y_direction,
        submit=args.submit,
        user=args.user,
        email=args.email,
        dryrun=False,
    )
    return None


def _convert_operetta(args) -> None:
    """Wrapper for convert_operetta()"""
    from blimp.preprocessing import convert_operetta

    convert_operetta(
        in_path=args.input_path,
        job_path=args.jobscript_path,
        image_format=args.output_format,
        n_batches=args.batch,
        save_metadata_files=True,
        mip=args.mip,
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
        default=os.getcwd(),
        help="Directory to save jobscripts (default = current working directory)",
        required=True,
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
        jobscript_path=args.jobscript_path,
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
        jobscript_path=args.jobscript_path,
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
        setup_parser = _get_setup_parser()
        setup_args = setup_parser.parse_args(sys.argv[sys.argv.index("setup") + 1 :])
        prepare_config(**vars(setup_args))
    else:
        args.func(args)

    return


if __name__ == "__main__":
    main()
