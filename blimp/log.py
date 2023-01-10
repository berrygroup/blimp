"""Setup global logging for blimp."""
import sys
import logging

#: Dict[int, int]: Mapping for logging verbosity to level
VERBOSITY_TO_LEVELS = {
    0: logging.NOTSET,
    1: logging.WARN,
    2: logging.INFO,
    3: logging.DEBUG,
}

#: Dict[int, int]: Mapping of logging level to verbosity
LEVELS_TO_VERBOSITY = {
    logging.NOTSET: 0,
    logging.WARN: 1,
    logging.INFO: 2,
    logging.DEBUG: 3,
}


def map_logging_verbosity(verbosity: int) -> int:
    """Maps logging verbosity to 'logging level', expected in the `logging`
    module.

    Parameters
    ----------
    verbosity
        logging verbosity
            * 0 = NOTSET
            * 1 = WARN
            * 2 = INFO
            * 3 = DEBUG

    Returns
    -------
    logging level

    Raises
    ------
    TypeError
        when `verbosity` doesn't have type int
    ValueError
        when `verbosity` is negative
    """
    if not isinstance(verbosity, int):
        raise TypeError('Argument "verbosity" must have type int.')
    if not verbosity >= 0:
        raise ValueError('Argument "verbosity" must be positive.')
    if verbosity >= len(VERBOSITY_TO_LEVELS):
        verbosity = len(VERBOSITY_TO_LEVELS) - 1
    verbosity_level = VERBOSITY_TO_LEVELS.get(verbosity)

    if verbosity_level is None:
        raise TypeError(f"Verbosity not mapped correctly, setting to WARN = {logging.WARN}")
        return logging.WARN
    else:
        return verbosity_level


def configure_logging(verbosity: int) -> None:
    """Configures a logger for command line applications. Two stream handlers
    are added:

        * "out" directs INFO & DEBUG messages to the
        standard output stream
        * "err" directs WARN, WARNING, ERROR, & CRITICAL
        messages to the standard error stream

    Parameters
    ----------
    verbosity
        logging verbosity
            * 0 = NOTSET
            * 1 = WARN
            * 2 = INFO
            * 3 = DEBUG
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s [[%(funcName)s @ %(pathname)s:%(lineno)d]]"

    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger = logging.getLogger()  # returns the root logger
    logger.setLevel(0)

    # remove any handlers loaded from imports
    logger.handlers.clear()

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.name = "err"
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARN)
    logger.addHandler(stderr_handler)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.name = "out"
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(map_logging_verbosity(verbosity))
    stdout_handler.addFilter(InfoFilter())
    logger.addHandler(stdout_handler)


class InfoFilter(logging.Filter):
    """Filter for allowing only INFO and DEBUG messages."""

    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)
