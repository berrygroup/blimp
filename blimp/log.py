"""
Copyright 2023 (C) University of New South Wales
Original author:
Scott Berry <scott.berry@unsw.edu.au>
"""
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

def map_logging_verbosity(verbosity : int) -> int:
    """
    Maps logging verbosity to 'logging' level

    Parameters
    ----------
    verbosity
        logging verbosity

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
    return VERBOSITY_TO_LEVELS.get(verbosity)


def configure_logging(verbosity : int):
    '''Configures the root logger for command line applications.
    Two stream handlers will be added to the logger:
        * "out" that will direct INFO & DEBUG messages to the standard output
        stream
        * "err" that will direct WARN, WARNING, ERROR, & CRITICAL messages to
        the standard error stream
    Note
    ----
    The level for individual loggers can be fine-tuned as follows (exemplified
    for the `tmlib` logger)::
        import logging
        logger = logging.getLogger('tmlib')
        logger.setLevel(logging.INFO)
    Warning
    -------
    Logging should only be configured once at the main entry point of the
    application!
    '''
    fmt = '%(asctime)s | %(levelname)-8s | %(message)s [[%(funcName)s @ %(pathname)s:%(lineno)d]]'

    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger = logging.getLogger()  # returns the root logger
    logger.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.name = 'err'
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARN)
    logger.addHandler(stderr_handler)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.name = 'out'
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(map_logging_verbosity(verbosity))
    stdout_handler.addFilter(InfoFilter())
    logger.addHandler(stdout_handler)


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


class Whitelist(logging.Filter):
    def __init__(self, *whitelist):
        self.whitelist = [logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any([f.filter(record) for f in self.whitelist])