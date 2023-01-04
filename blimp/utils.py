import os
import logging

#: Dict[int, int]: Mapping for logging verbosity to level
VERBOSITY_TO_LEVELS = {
    0: logging.NOTSET,
    1: logging.WARN,  # For simplicity. Includes ERROR, CRITICAL
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

def init_logging(level: int = logging.INFO) -> None:
    """
    Setup logging for BLIMP

    Parameters
    ----------
    level
        logging level.
        See `logging levels <https://docs.python.org/3/library/logging.html#logging-levels>`_
        for a list of levels.
    """
    import warnings

    fmt = '%(asctime)s | %(process)5d/%(threadName)-12s| %(levelname)-8s| %(message)s [[%(name)s @ %(pathname)s:%(lineno)d]]'
    datefmt = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(level=level,format=fmt,datefmt=datefmt)
    logging.getLogger().setLevel(level)
    warnings.filterwarnings("ignore", message="Failed to parse XML for the provided file.")
