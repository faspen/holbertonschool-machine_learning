#!/usr/bin/env python3
"""RNN Encoder"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encode for machine translastion"""

    def __init__(self, vocab, embedding, units, batch):
        """RNNEncoder class constructor"""
        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Initialize hidden states of RNN Cell to zeros"""
        init = tf.keras.initializers.Zeros()
        values = init(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """Returns outputs and hiddens state of the encoder"""
        inp = self.embedding(x)
        outputs, hidden = self.gru(inp, initial_state=initial)
        return outputs, hidden
