""" module to define the generative text RNN based model """

import config as cfg
import tensorflow as tf

class DefineXSModel():
    """ class to make encoder decoder model for generative sentence completion """

    def __init__(self) -> None:
        pass

    def encoder(self) -> tf.Tensor:
        """ 
        function to define the encoder of a given sentence
        into a vector space that holds the sentence context 
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=cfg.MAX_INPUT_VECTOR_LENGTH))
        # model.add(tf.keras.layers.SimpleRNN(cfg.NUMBER_OF_RNN_NEURONS))


    def decoder(self) -> list:
        """ 
        function to define the decoder of a given vector
        into a sentence that completes the provided sentence context 
        """
        pass

    def define_model(self) -> tf.keras.Model:
        """ function to define the complete encoder decoder problem """
        pass
