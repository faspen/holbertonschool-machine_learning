#!/usr/bin/env python3
"""Self Attention module"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation"""

    def __init__(self, units):
        """Self attention constructor"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Perform the embedding"""
        expanded = tf.expand_dims(input=s_prev, axis=1)
        inputs = self.U(expanded)
        hidden = self.W(hidden_states)
        result = self.V(tf.nn.tanh(inputs + hidden))

        weights = tf.nn.softmax(result, axis=1)
        context = weights * expanded
        context = tf.reduce_sum(context, axis=1)

        return context, weights
