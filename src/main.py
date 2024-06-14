""" Application Entry Point """

import os
import json as js
import pickle as pk
import logging as lg
import config as cfg
import tensorflow as tf

from typing import Generator, Set
from models.rnn import DefineSModel
from utils.preprocess import PreProcess
from helpers.data_loader import DataLoader
from helpers.data_splitter import FeatureExtractor

# SELECT THE DATASET
# =============================================================================================
current_dataset_type = cfg.DatasetType.SMALL                    # SMALL/MEDIUM/LARGE/VERY_LARGE
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

    # STEP 1:
    # loading the dataset based on the type mentioned and then preprocess it.

    if not os.path.exists(tensor_pickle_file) or cfg.RETRAIN_MODEL:
        for dataset_tensor in load_and_process_data(current_dataset_type):
            with open(tensor_pickle_file, "wb") as tensor_pickle:
                pk.dump(dataset_tensor, tensor_pickle)
    else:
        lg.warning("Tensors for the given data already exist. Using the existing values")

    # STEP 2:
    # defining an encoder and decoder model based on the selected data the architecture is chosen

    if not os.path.exists(vocab_json_file) or cfg.RETRAIN_MODEL:
        with open(vocab_json_file, "r", encoding="utf-8") as vocab_json:
            if current_dataset_type == cfg.DatasetType.SMALL:
                current_model = DefineSModel(js.load(vocab_json)).run()
    else:
        lg.warning("Vocabulary JSON for given dataset already exist. Using the existing vocab")

    # STEP 3:
    # loading preprocessed data and training the model.

    if not os.path.exists(trained_model_path) or cfg.RETRAIN_MODEL:
        with open(tensor_pickle_file, "rb") as tensor_pickle:
            try:
                while True:
                    dataset_tensors = pk.load(tensor_pickle)
                    lg.debug("tensors: %s", dataset_tensors)
                    features, labels = FeatureExtractor(dataset_tensors).run()
                    current_model.fit(features, labels, epochs=cfg.TOTAL_EPOCHS)
                    current_model.save(trained_model_path)
            except EOFError:
                pass
    else:
        lg.waring("A Trained model for the given dataset is found. Using the found model")

    # loaded_model = tf.keras.models.load_model(trained_model_path)
    # lg.debug("predict: %s", loaded_model.predict(features))

    # STEP 4:
    # plot results

    # STEP 5:
    # test model
