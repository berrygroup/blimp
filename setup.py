from distutils.core import setup

setup(
    name="blimp",
    version="0.1.0",
    author="berrygroup",
    author_email="scottdberry@gmail.com",
    packages=["blimp", "blimp.processing", "blimp.preprocessing"],
    scripts=[],
    url="https://github.com/berrygroup/blimp",
    license="LICENSE.txt",
    description="Berry lab main image processing utilities",
    long_description=open("README.md").read(),
    install_requires=[],
)
