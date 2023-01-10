from pathlib import Path

from setuptools import setup, find_packages

try:
    from blimp import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "berrygroup"
    __email__ = "scott.berry@unsw.edu.au"
    __version__ = "0.1.0"


setup(
    name="blimp",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description="Berry lab image processing utilities",
    long_description=Path("README.rst").read_text("utf-8"),
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/berrygroup/blimp",
    #    download_url="https://pypi.org/project/blimp/",
    project_urls={
        #        "Documentation": "https://blimp.readthedocs.io/en/stable",
        "Source Code": "https://github.com/berrygroup/blimp",
    },
    zip_safe=False,
    license="LICENSE.txt",
    platforms=["Linux", "MacOSX"],
    packages=find_packages(),
    package_dir={"blimp": "blimp"},
    include_package_data=True,
    extras_require=dict(
        dev=["pre-commit>=2.9.0"],
        test=[
            "tox>=3.20.1",
            "pytest",
            "pytest-cov",
            "pytest-dependency",
        ],
        docs=[
            l.strip()
            for l in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            if not l.startswith("-r")
        ],
    ),
    install_requires=[line.strip() for line in Path("requirements.txt").read_text("utf-8").splitlines()],
    #    entry_points={
    #        "console_scripts": ["campa=campa.cli.main:CAMPA"],
    #    },
)
