""" module to store all the constants, paths and values in one place """

import os

from enum import Enum

# =============================================================
# paths
# =============================================================

TEST_DATA_PATH = os.path.join("data", "testing")
TRAINING_DATA_PATH = os.path.join("data", "training")

# =============================================================
# data constants
# =============================================================

class DatasetType(Enum):
    """ class for storing file names of different datasets """
    SMALL = {"text": "small_dataset.txt", "pickle": "small_dataset_pickle.pkl"}
    MEDIUM = {"text": "medium_dataset.txt", "pickle": "medium_dataset_pickle.pkl"}
    LARGE = {"text": "large_dataset.txt", "pickle": "large_dataset_pickle.pkl"}
    VERY_LARGE = {"text": "very_large_dataset.txt", "pickle": "very_large_dataset_pickle.pkl"}

CHUNK_SIZE = 10000

# =============================================================
# model constants
# =============================================================

# vector shape constants
MAX_INPUT_VECTOR_LENGTH = 35
MAX_OUTPUT_EMBEDDED_LENGTH = 30

# model architecture constants
NUMBER_OF_RNN_NEURONS = 20
