#!/usr/bin/env python3
"""Transformer encoder block"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Encoder block class initializer"""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def split_last(self, x, batch_size):
        """Split last dimension"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training, mask=None):
        """Returns the blocks output"""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        res = self.layernorm1(x + attention)

        hidden = self.dense_hidden(res)
        output = self.dense_output(hidden)

        drop = self.dropout2(output, training=training)
        output = self.layernorm2(res + drop)

        return output
