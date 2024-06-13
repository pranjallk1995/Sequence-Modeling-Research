""" Application Entry Point """

import logging as lg
import config as cfg
import tensorflow as tf

from typing import Generator
from utils.preprocess import PreProcess
from helpers.data_loader import DataLoader

def load_and_process_data(dataset_type: cfg.DatasetType) -> Generator[tf.Tensor, None, None]:
    """ function to load any type of dataset and process it """

    data_chunk = DataLoader(dataset_type).run()
    for data_chunk_id, data_chunk_value in enumerate(data_chunk):
        lg.debug("loading dataset chunk %d from %s", data_chunk_id, dataset_type.value)
        lg.debug("data: %s", data_chunk_value)
        yield PreProcess(dataset=data_chunk_value).run()

if __name__ == "__main__":

    # setting the logging level to display in console as INFO.
    lg.basicConfig(level=lg.DEBUG, filename="run.log", filemode="w")

    # defining an encoder and decoder model
    # model = 

    # loading the dataset based on the type mentioned asynchronously then preprocess it.
    for dataset_tensor in load_and_process_data(cfg.DatasetType.SMALL):
        lg.debug("dataset tensor: %s", dataset_tensor)

    # train model

    # plot results

    # test model
