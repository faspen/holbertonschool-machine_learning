#!/usr/bin/env python3
"""Transformer Decoder"""


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Create the decoder for transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Decoder class initializer"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Creates the decoder block"""
        input_seq_len = x.shape[1]
        embed = self.embedding(x)
        embed *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embed += self.positional_encoding[:input_seq_len]
        result = self.dropout(embed, training=training)

        for i in range(self.N):
            result = self.blocks[i](result, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        return result
