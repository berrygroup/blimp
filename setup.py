from pathlib import Path

from setuptools import setup, find_packages

try:
    from blimp import __email__, __author__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "Scott Berry"
    __email__ = "scott.berry@unsw.edu.au"


setup(
    name="blimp",
    # Do not add an explicit version= here: it silently overrides
    # use_scm_version, which is how this package came to report 0.1.0 while
    # tagged v0.4.0.
    use_scm_version=True,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description="Berry lab image processing utilities",
    long_description=Path("README.md").read_text("utf-8"),
    long_description_content_type="text/markdown; charset=UTF-8",
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
    install_requires=[
        line.strip()
        for line in Path("requirements.txt").read_text("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ],
    entry_points={
        "console_scripts": ["blimp=blimp.cli.main:main"],
    },
)
