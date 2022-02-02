#!/usr/bin/env python3
"""RNN Decoder Task"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """RNN Decoder initializer"""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Returns the y and s tensors"""
        items = s_prev.get_shape().as_list()[1]
        tmp_func = SelfAttention(items)
        context, weight = tmp_func(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)

        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)

        return y, s
