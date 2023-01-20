import os

import pytest

# import logging


@pytest.fixture(autouse=True)
def _set_config():
    """
    set EXPERIMENT_DIR and BASE_DATA_DIR in blimp config
    """
    from blimp.constants import SCRIPTS_DIR, blimp_config

    # FIXME: this call to configure_logging seems to inhibit logging during testing?
    # configure_logging(logging.DEBUG)  # NOTE: this will silence some warnings of 3rd party packages
    # TODO: implement test data download
    # from blimp.data._download_data import load_test_data
    # ensure that test data is downloaded
    # load_test_data()

    blimp_config.EXPERIMENT_DIR = os.path.join(SCRIPTS_DIR, "tests", "_experiments")
    blimp_config.BASE_DATA_DIR = os.path.join(SCRIPTS_DIR, "tests", "_data")
    blimp_config.add_data_config("TestData", os.path.join(SCRIPTS_DIR, "tests/_data/TestData_constants.py"))

    print("BLIMP CONFIG:")
    print(blimp_config)
