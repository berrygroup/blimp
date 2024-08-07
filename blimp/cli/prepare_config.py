#!python
from pathlib import Path
import os

from blimp.constants import blimp_config


def prepare_config(quiet=False):
    def get_yn_input():
        r = input()
        while r.lower() not in ("y", "n"):
            r = input("Unknown value. Either input y or n:\n")
        if r.lower() == "y":
            return True
        return False

    def get_path_input(name, current_value=None):
        print(f"The current {name} is {current_value}. Would you like to change it? (y/n)")
        if not get_yn_input():
            print(f"Leaving {name} unchanged.")
            return current_value
        else:
            p = input(f"Please enter the full path to {name}:\n")
            p = os.path.abspath(p.strip())
            if not os.path.isdir(p):
                print(f"{p} is not a valid directory")
                return get_path_input(name)
            return p

    if blimp_config.config_fname is not None:
        print(f"Found existing config in {blimp_config.config_fname}")
        config_fname = blimp_config.config_fname
    else:
        config_fname = Path.home() / ".config" / "blimp" / "blimp.ini"
        print(f"No config found. Creating default config in {config_fname}.")
        # read config file from scripts_dir (parent dir of dir that this file is in)
        example_config_path = Path(__file__).resolve().parent.parent / "blimp.ini.example"
        blimp_config.config_fname = example_config_path
        blimp_config.write(config_fname)

    # check test data
    cur_test_data = blimp_config.data_configs.get("ExampleData", None)
    if cur_test_data in ("", None) or not Path(cur_test_data).is_file():
        print("setting up ExampleData config")
        # add test data
        blimp_config.add_data_config(
            "ExampleData",
            str(Path(__file__).resolve().parent.parent.parent / "notebooks" / "ExampleData_constants.py"),
        )
        blimp_config.write()

    if not quiet:
        print("Would you like to configure your blimp.ini config now? (y/n)")
        if get_yn_input():
            # read config file
            blimp_config.EXPERIMENT_DIR = get_path_input("EXPERIMENT_DIR", blimp_config.EXPERIMENT_DIR)
            blimp_config.BASE_DATA_DIR = get_path_input("BASE_DATA_DIR", blimp_config.BASE_DATA_DIR)
            blimp_config.write()
        else:
            print("Exiting without setting up blimp.ini")

    # print currently registered data config files
    print(f"Currently registered data configs in {blimp_config.config_fname}:")
    for key, val in blimp_config.data_configs.items():
        print(f"\t{key}: {val}")
    print(f"To change or add data configs, please edit {blimp_config.config_fname} directly.")
