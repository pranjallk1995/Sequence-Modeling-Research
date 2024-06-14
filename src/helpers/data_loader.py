""" module to load the given type of dataset into application's memory """

import os
import pandas as pd
import logging as lg
import config as cfg

from typing import Generator

class DataLoader():
    """ class to load the given dataset into memory """

    def __init__(self, dataset_type: cfg.DatasetType) -> None:
        self.dataset_type = dataset_type

    def load_data(self) -> Generator[pd.DataFrame, None, None]:
        """ function to load the given dataset into memeory in chuncks and return it """

        lg.info("loading dataset from %s", self.dataset_type.value["text"])
        data_reader = pd.read_csv(
            os.path.join(cfg.TRAINING_DATA_PATH, self.dataset_type.value["text"]),
            sep="\0", header=None, names=["Sentence"], chunksize=cfg.CHUNK_SIZE
        )
        for data_chunk in data_reader:
            yield data_chunk

    def run(self) -> Generator[pd.DataFrame, None, None]:
        """ module entry point """
        return self.load_data()
