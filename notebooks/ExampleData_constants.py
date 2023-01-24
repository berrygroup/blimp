# Constants for TestData dataset
import os

from blimp.constants import blimp_config

# --- dataset specific constants ---
DATA_DIR = blimp_config.BASE_DATA_DIR
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
RESOURCES_DIR = os.path.join(DATA_DIR, "resources")
RESULTS_DIR = os.path.join(blimp_config.EXPERIMENT_DIR, "example_results")
