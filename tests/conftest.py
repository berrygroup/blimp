import os

import pytest


@pytest.fixture(autouse=True)
def _set_config():
    """
    set EXPERIMENT_DIR and BASE_DATA_DIR in blimp config
    """
    from blimp.log import configure_logging
    from blimp.constants import SCRIPTS_DIR, blimp_config

    # from blimp.data._download_data import load_test_data

    configure_logging()  # NOTE: this will silence some warnings of 3rd party packages
    # ensure that test data is downloaded
    # load_test_data()

    blimp_config.EXPERIMENT_DIR = os.path.join(SCRIPTS_DIR, "tests", "_experiments")
    blimp_config.BASE_DATA_DIR = os.path.join(SCRIPTS_DIR, "tests", "_data")
    blimp_config.add_data_config("TestData", os.path.join(SCRIPTS_DIR, "tests/_data/TestData_constants.py"))

    print("BLIMP CONFIG:")
    print(blimp_config)
