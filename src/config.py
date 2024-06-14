""" module to store all the constants, paths and values in one place """

import os

from enum import Enum

# =============================================================
# paths
# =============================================================

TEST_DATA_PATH = os.path.join("data", "testing")
TRAINING_DATA_PATH = os.path.join("data", "training")
TRAINED_MODEL_PATH = os.path.join("models")

# =============================================================
# data constants
# =============================================================

class DatasetType(Enum):
    """ class for storing file names of different datasets """
    SMALL = {
        "text": "small_dataset.txt",
        "pickle": "small_dataset_pickle.pkl",
        "json": "small_dataset_vocab.json",
        "model": "s_model.keras",
        "fig": "s_model.html"
    }
    MEDIUM = {
        "text": "medium_dataset.txt",
        "pickle": "medium_dataset_pickle.pkl",
        "json": "medium_dataset_vocab.json",
        "model": "m_model.keras",
        "fig": "m_model.html"
    }
    LARGE = {
        "text": "large_dataset.txt",
        "pickle": "large_dataset_pickle.pkl",
        "json": "large_dataset_vocab.json",
        "model": "l_model.keras",
        "fig": "l_model.html"
    }
    VERY_LARGE = {
        "text": "very_large_dataset.txt",
        "pickle": "very_large_dataset_pickle.pkl",
        "json": "very_large_dataset_vocab.json",
        "model": "vl_model.keras",
        "fig": "vl_model.html"
    }

PUNCTUATIONS = ["?", "!", "'", ",", ".", ";", "\""]
CHUNK_SIZE = 10000

# =============================================================
# model constants
# =============================================================

RETRAIN_MODEL = True

# vector shape constants
MAX_INPUT_VECTOR_LENGTH = 18
MAX_OUTPUT_EMBEDDED_LENGTH = 10
MAX_FEATURE_LENGTH = 9

# model architecture constants
NUMBER_OF_RNN_NEURONS = 8
LEARNING_RATE = 0.01
TOTAL_EPOCHS = 10
