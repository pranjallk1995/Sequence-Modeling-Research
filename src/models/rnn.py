""" module to define the generative text RNN based model """

import config as cfg
import logging as lg
import tensorflow as tf

class DefineSModel():
    """ class to make encoder decoder model for generative sentence completion """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.encoder_input = None
        self.encoder_state = None
        self.encoder_output = None
        self.final_output = None

    def encoder(self) -> None:
        """ 
        function to define the encoder of a given sentence
        into a vector space that holds the sentence context 
        """

        self.encoder_input = tf.keras.layers.Input(
            shape=(cfg.MAX_FEATURE_LENGTH,), name="encoder_input"
        )

        encoder_embedding = tf.keras.layers.Embedding(
            name="encoder_embedding",
            input_dim=self.vocab_size, output_dim=cfg.MAX_OUTPUT_EMBEDDED_LENGTH
        )(self.encoder_input)

        self.encoder_output, self.encoder_state = tf.keras.layers.SimpleRNN(
            units=cfg.NUMBER_OF_RNN_NEURONS, return_state=True, name="encoder_layer_1"
        )(encoder_embedding)

    def decoder(self) -> None:
        """ 
        function to define the decoder of a given vector
        into a sentence that completes the provided sentence context 
        """

        decoder_embedding = tf.keras.layers.Embedding(
            name="decoder_embedding",
            input_dim=self.vocab_size, output_dim=cfg.MAX_OUTPUT_EMBEDDED_LENGTH
        )(self.encoder_output)

        decoder_output = tf.keras.layers.SimpleRNN(
            units=cfg.NUMBER_OF_RNN_NEURONS, name="decoder_layer_1"
        )(decoder_embedding, initial_state=self.encoder_state)

        self.final_output = tf.keras.layers.Dense(
            units=cfg.MAX_INPUT_VECTOR_LENGTH-cfg.MAX_FEATURE_LENGTH, activation="sigmoid"
        )(decoder_output)

    def run(self) -> tf.keras.Model:
        """ module entry point """
        self.encoder()
        self.decoder()
        model = tf.keras.Model(inputs=self.encoder_input, outputs=self.final_output)
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
            loss=tf.keras.losses.MeanSquaredError()
        )
        return model
