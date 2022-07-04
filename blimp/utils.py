import os
import logging

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

    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    warnings.filterwarnings("ignore", message="Failed to parse XML for the provided file.")
