#!/usr/bin/env python3
"""Self Attention module"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation"""

    def __init__(self, units):
        """Self attention constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Perform the embedding"""
        inputs = self.W(tf.expand_dims(s_prev, 1))
        hidden = self.U(hidden_states)
        result = self.V(tf.nn.tanh(inputs + hidden))

        weights = tf.nn.softmax(result, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
