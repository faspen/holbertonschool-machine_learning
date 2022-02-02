#!/usr/bin/env python3
"""Multi head attention"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """MHA class initializer"""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_last(self, x, batch_size):
        """Split last dimension"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Perform the multi head attention"""
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_last(q, batch_size)
        k = self.split_last(k, batch_size)
        v = self.split_last(v, batch_size)

        scaled, weights = sdp_attention(q, k, v, mask)
        scaled = tf.transpose(scaled, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled, (batch_size, -1, self.dm))
        output = self.linear(concat)

        return output, weights
