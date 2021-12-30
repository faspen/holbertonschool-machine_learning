#!/usr/bin/env python3
"""Transformer decoder block"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Creates encoder block for transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Decoderblock class initializer"""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Create the decoder block"""
        att1, weights1 = self.mha1(x, x, x, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        norm1 = self.layernorm1(att1 + x)

        att2, weights2 = self.mha2(norm1, encoder_output, encoder_output,
                                   padding_mask)
        att2 = self.dropout2(att2, training=training)
        norm2 = self.layernorm2(att2 + norm1)

        den1 = self.dense_hidden(norm2)
        den2 = self.dense_output(den1)
        drop = self.dropout3(den2, training=training)
        norm3 = self.layernorm3(norm2 + drop)

        return norm3
