__author__ = __maintainer__ = "Scott Berry"
__email__ = "scott.berry@unsw.edu.au"

# The version is owned by setuptools_scm, which derives it from the git tags at
# build time and records it in the installed package metadata. Reading it back
# from that metadata keeps a single source of truth: there is no hardcoded
# number here to drift out of step with the tags.
#
# PackageNotFoundError means blimp is being imported without being installed
# (e.g. from a source checkout on sys.path, or by setup.py during its own
# build). There is no metadata to read in that case, so fall back to a marker
# that is a valid PEP 440 version but is obviously not a release.
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Retained for backwards compatibility: __full_version__ used to carry the
# local part of the version separately. setuptools_scm now includes it in
# __version__ directly, so the two are the same.
__full_version__ = __version__

del _version, PackageNotFoundError
