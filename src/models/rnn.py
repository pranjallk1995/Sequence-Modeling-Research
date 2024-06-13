""" module to define the generative text RNN based model """

import config as cfg
import tensorflow as tf

from typing import Set

class DefineXSModel():
    """ class to make encoder decoder model for generative sentence completion """

    def __init__(self, dataset_vocab: Set[str]) -> None:
        self.vocab = dataset_vocab
        self.encoder_input = None
        self.encoder_output = None

    def encoder(self) -> tf.Tensor:
        """ 
        function to define the encoder of a given sentence
        into a vector space that holds the sentence context 
        """
        self.encoder_input = tf.keras.layers.Input(
            shape=(cfg.MAX_INPUT_VECTOR_LENGTH,), name="encoder_input"
        )

        encoder_embedding = tf.keras.layers.Embedding(
            input_length=cfg.MAX_INPUT_VECTOR_LENGTH, name="encoder_embedding",
            input_dim=len(self.vocab), output_dim=cfg.MAX_OUTPUT_EMBEDDED_LENGTH
        )(self.encoder_input)

        self.encoder_output = tf.keras.layers.SimpleRNN(
            cfg.NUMBER_OF_RNN_NEURONS
        )(encoder_embedding)

    def decoder(self) -> list:
        """ 
        function to define the decoder of a given vector
        into a sentence that completes the provided sentence context 
        """
        pass

    def run(self) -> tf.keras.Model:
        """ module entry point """
        pass
