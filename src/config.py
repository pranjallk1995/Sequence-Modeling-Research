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
    SMALL = {
        "text": "small_dataset.txt",
        "pickle": "small_dataset_pickle.pkl",
        "json": "small_dataset_vocab.json"
    }
    MEDIUM = {
        "text": "medium_dataset.txt",
        "pickle": "medium_dataset_pickle.pkl",
        "json": "medium_dataset_vocab.json"
    }
    LARGE = {
        "text": "large_dataset.txt",
        "pickle": "large_dataset_pickle.pkl",
        "json": "large_dataset_vocab.json"
    }
    VERY_LARGE = {
        "text": "very_large_dataset.txt",
        "pickle": "very_large_dataset_pickle.pkl",
        "json": "very_large_dataset_vocab.json"
    }

PUNCTUATIONS = ["?", "!", "'", ",", ".", ";", "\""]
CHUNK_SIZE = 4

# =============================================================
# model constants
# =============================================================

# vector shape constants
MAX_INPUT_VECTOR_LENGTH = 18
MAX_OUTPUT_EMBEDDED_LENGTH = 10

# model architecture constants
NUMBER_OF_RNN_NEURONS = 10
