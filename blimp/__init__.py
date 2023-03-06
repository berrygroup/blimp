__author__ = __maintainer__ = "Scott Berry"
__email__ = "scott.berry@unsw.edu.au"
__version__ = "0.1.0"

from importlib.metadata import version

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse
