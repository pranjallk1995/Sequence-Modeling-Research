""" Application Entry Point """

import logging as lg
import config as cfg

from helpers.data_loader import DataLoader

def load_and_process_data(dataset_type: cfg.DatasetType):
    """ asynchronous function to load any type of dataset and process it """
    data_chunk = DataLoader(dataset_type).load_data()
    for chunk_id, chunck_data in enumerate(data_chunk):
        lg.debug("loading dataset chunk %d from %s", chunk_id, dataset_type.value)
        print(chunck_data)

if __name__ == "__main__":

    # setting the logging level to display in console as INFO.
    lg.basicConfig(level=lg.INFO)

    # loading the dataset based on the type mentioned asynchronously then preprocess it.
    load_and_process_data(cfg.DatasetType.SMALL)

    # define model

    # train model

    # plot results

    # test model
