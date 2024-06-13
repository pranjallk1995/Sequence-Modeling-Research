""" module to store all the constants, paths and values in one place """

import os

from enum import Enum

# ========================================================
# paths
# ========================================================

TEST_DATA_PATH = os.path.join("data", "testing")
TRAINING_DATA_PATH = os.path.join("data", "training")

# ========================================================
# data constants
# ========================================================

class DatasetType(Enum):
    """ class for storing file names of different datasets """
    SMALL = "small_dataset.txt"
    MEDIUM = "medium_dataset.txt"
    LARGE = "large_dataset.txt"
    VERY_LARGE = "very_large_dataset.txt"

CHUNK_SIZE = 3
