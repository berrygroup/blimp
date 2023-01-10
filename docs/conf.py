# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# sys.path.insert(0, os.path.abspath('.'))
from pathlib import Path
from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # this way, we don't have to install blimp
import blimp  # noqa: E402

# -- Project information -----------------------------------------------------

project = "blimp"
author = blimp.__author__
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001

# The full version, including alpha/beta/rc tags
release = blimp.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    dask=("https://docs.dask.org/en/latest/", None),
    skimage=("https://scikit-image.org/docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
    tensorflow=(
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# spelling
spelling_lang = "en_GB"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_show_suggestions = False
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
    "docs.source.utils.ModnameFilter",
    "docs.source.utils.MDLinkFilter",
    "docs.source.utils.SignatureFilter",
]
# spelling_exclude_patterns = ["*<autosummary>*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
autosummary_generate = True
add_module_names = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
