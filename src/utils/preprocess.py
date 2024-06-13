""" module to preprecess data before passing to the model """

import pandas as pd
import config as cfg
import logging as lg
import tensorflow as tf

class PreProcess():
    """ class to perform preprocessing on the dataset """

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.vocab = None
        self.dataset_tensor = None

    def normalize_dataset(self) -> tf.Tensor:
        """ function to normalize the given dataset """

        lg.info("Performing normalization of the dataset chunk")
        normalization_layer = tf.keras.layers.Normalization(axis=0)
        normalization_layer.adapt(self.dataset_tensor)
        return normalization_layer(self.dataset_tensor)


    def vectorize_dataset(self) -> tf.Tensor:
        """ function to perform vectorization of the dataset """

        lg.info("Performing vectorization of the dataset chunk")
        vectorization_layer = tf.keras.layers.TextVectorization(
            output_mode="int", output_sequence_length=cfg.MAX_INPUT_VECTOR_LENGTH,
            name="vectorize", standardize="lower_and_strip_punctuation"
        )
        vectorization_layer.adapt(self.dataset)
        self.vocab = vectorization_layer.get_vocabulary()
        lg.debug("vocab: %s", self.vocab)
        return vectorization_layer(self.dataset)

    def run(self) -> tf.Tensor:
        """ module entry point """
        self.dataset_tensor = self.vectorize_dataset()
        if self.dataset_tensor is not None:
            lg.debug("tensor %s", self.dataset_tensor)
            return self.normalize_dataset()
        else:
            lg.error("Could not vectorize the dataset")
            return None
