""" module to preprecess data before passing to the model """

import pandas as pd
import logging as lg
import tensorflow as tf

class PreProcess():
    """ class to perform preprocessing on the dataset """

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.vocab = []

    def vectorize_dataset(self) -> tf.Tensor:
        """ function to perform vectorization of the dataset """

        lg.info("Performing vectorization of the dataset chunk")
        vectorization_layer = tf.keras.layers.TextVectorization(
            output_mode="int", name="vectorize", standardize="lower_and_strip_punctuation"
        )
        vectorization_layer.adapt(self.dataset)
        self.vocab += vectorization_layer.get_vocabulary()
        lg.debug("vocab: %s", self.vocab)

        return vectorization_layer(self.dataset)
