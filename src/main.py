""" Application Entry Point """

import os
import json as js
import pickle as pk
import logging as lg
import config as cfg
import tensorflow as tf

from typing import Generator
from models.rnn import DefineSModel
from utils.plotting import Plotting
from utils.preprocess import PreProcess
from helpers.data_loader import DataLoader
from helpers.data_splitter import FeatureExtractor

# SELECT THE DATASET
# =============================================================================================
current_dataset_type = cfg.DatasetType.SMALL                    # SMALL/MEDIUM/LARGE/VERY_LARGE
current_dataset_vocab_size = None
# =============================================================================================

def load_and_process_data(dataset_type: cfg.DatasetType) -> Generator[tf.Tensor, None, None]:
    """ function to load any type of dataset and process it """

    data_chunk = DataLoader(dataset_type).run()
    preprocess_obj = PreProcess()
    for data_chunk_id, data_chunk_value in enumerate(data_chunk):
        lg.debug("loading dataset chunk %d from %s", data_chunk_id, dataset_type.value["text"])
        lg.debug("data: %s", data_chunk_value)
        yield preprocess_obj.run(data_chunk_id, data_chunk_value, dataset_type)

if __name__ == "__main__":

    # setting the logging level to display.
    lg.basicConfig(level=lg.DEBUG, filename="run.log", filemode="w")

    tensor_pickle_file = os.path.join(cfg.TRAINING_DATA_PATH, current_dataset_type.value["pickle"])
    vocab_json_file = os.path.join(cfg.TRAINING_DATA_PATH, current_dataset_type.value["json"])
    trained_model_path = os.path.join(cfg.TRAINED_MODEL_PATH, current_dataset_type.value["model"])
    model_plot_path = os.path.join(cfg.TRAINED_MODEL_PATH, current_dataset_type.value["fig"])

    # STEP 1:
    # loading the dataset based on the type mentioned and then preprocess it.
    # it stores the tensors in a pickle file and vocabulary in a JSON file.

    if not os.path.exists(tensor_pickle_file) or cfg.RETRAIN_MODEL:
        for dataset_tensor in load_and_process_data(current_dataset_type):
            with open(tensor_pickle_file, "wb") as tensor_pickle:
                pk.dump(dataset_tensor, tensor_pickle)
    else:
        lg.warning("Tensors for the given data already exist. Using the existing values")

    # STEP 2:
    # defining an encoder and decoder model based on the selected data the architecture is chosen

    if os.path.exists(vocab_json_file):
        with open(vocab_json_file, "r", encoding="utf-8") as vocab_json:
            if current_dataset_type == cfg.DatasetType.SMALL:
                current_dataset_vocab_size = len(js.load(vocab_json))
                current_model = DefineSModel(current_dataset_vocab_size).run()
    else:
        lg.error("Vocabulary JSON for dataset not found in %s", vocab_json_file)

    # STEP 3:
    # loading preprocessed data and training the model.

    training_history = None
    if not os.path.exists(trained_model_path) or cfg.RETRAIN_MODEL:
        with open(tensor_pickle_file, "rb") as tensor_pickle:
            try:
                while True:
                    # make tensor values between 0 and 1
                    dataset_tensors = pk.load(tensor_pickle)/current_dataset_vocab_size
                    lg.debug("tensors: %s", dataset_tensors)
                    features, labels = FeatureExtractor(dataset_tensors).run()
                    training_history = current_model.fit(features, labels, epochs=cfg.TOTAL_EPOCHS)
            except EOFError:
                pass
            current_model.save(trained_model_path)
    else:
        lg.waring("A Trained model for the given dataset is found. Using the found model")

    # STEP 4:
    # plot results
    if training_history is not None:
        Plotting(training_history, model_plot_path).plot()

    # STEP 5:
    # test model
