__author__ = __maintainer__ = "Scott Berry"
__email__ = "scott.berry@unsw.edu.au"

# Version comes from the git tags via setuptools_scm; the fallback covers
# import without installation, where there is no metadata to read.
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Retained for backwards compatibility.
__full_version__ = __version__

del _version, PackageNotFoundError
