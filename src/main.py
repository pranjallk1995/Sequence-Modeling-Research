""" Application Entry Point """

import os
import pickle as pk
import logging as lg
import config as cfg
import tensorflow as tf

from typing import Generator, Set
from models.rnn import DefineXSModel
from utils.preprocess import PreProcess
from helpers.data_loader import DataLoader

# SELECT THE DATASET
# =============================================================================================
current_dataset_type = cfg.DatasetType.SMALL                    # SMALL/MEDIUM/LARGE/VERY_LARGE
# =============================================================================================

def load_and_process_data(
        dataset_type: cfg.DatasetType, vocab: Set[str]
    ) -> Generator[tf.Tensor, None, None]:
    """ function to load any type of dataset and process it """

    data_chunk = DataLoader(dataset_type).run()
    for data_chunk_id, data_chunk_value in enumerate(data_chunk):
        lg.debug("loading dataset chunk %d from %s", data_chunk_id, dataset_type.value["text"])
        lg.debug("data: %s", data_chunk_value)
        yield PreProcess(dataset_id=data_chunk_id, dataset=data_chunk_value, vocab=vocab).run()

if __name__ == "__main__":

    # setting the logging level to display in console as INFO.
    lg.basicConfig(level=lg.INFO, filename="run.log", filemode="w")

    # STEP 1:
    # loading the dataset based on the type mentioned and then preprocess it.
    
    current_vocab = set()
    tensor_pickle_file = os.path.join(cfg.TRAINING_DATA_PATH, current_dataset_type.value["pickle"])
    
    if not os.path.exists(tensor_pickle_file):
        for vocab_size, dataset_tensor in load_and_process_data(current_dataset_type, current_vocab):
            lg.debug("vocab size: %d", vocab_size)
            lg.debug("dataset tensor: %s", dataset_tensor)
            with open(tensor_pickle_file, "wb") as tensor_pickle:
                pk.dump(dataset_tensor, tensor_pickle)
    else:
        lg.warning("Tensors for the given data already exist. Using the existing values")

    # STEP 2:
    # defining an encoder and decoder model based on the selected data the architecture is chosen

    model = DefineXSModel.run()

    # STEP 3:
    # loading preprocessed data and training the model.

    with open(tensor_pickle_file, "rb") as tensor_pickle:
        try:
            while True:
                dataset_tensors = pk.load(tensor_pickle)
                # train model on this batch od data
        except EOFError:
            pass

    # STEP 4:
    # plot results

    # STEP 5:
    # test model
