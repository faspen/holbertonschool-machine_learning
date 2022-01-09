#!/usr/bin/env python3
"""Transformer network"""


import tensorflow as tf
import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention"""
    mult = tf.matmul(Q, K, transpose_b=True)
    cst_K = tf.cast(tf.shape(K)[-1], tf.float32)
    square = mult / tf.math.sqrt(cst_K)

    if mask is not None:
        square += (mask * -1e9)

    weights = tf.nn.softmax(square, axis=-1)
    outputs = tf.matmul(weights, V)

    return outputs, weights


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


def positional_encoding(max_seq_len, dm):
    """Calculates the positionaal encoding for a transformer"""
    pos_embeddings = np.zeros((max_seq_len, dm))

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            power = np.exp(j * -np.log(10000.0) / dm)
            pos_embeddings[i, j] = (np.sin(i * power))
            pos_embeddings[i, j + 1] = (np.cos(i * power))

    return pos_embeddings


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


class Encoder(tf.keras.layers.Layer):
    """Create the encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Encoder class initializer"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Creates the encoder block"""
        input_seq_len = x.shape[1]
        embed = self.embedding(x)
        embed *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embed += self.positional_encoding[:input_seq_len]
        result = self.dropout(embed, training=training)

        for i in range(self.N):
            result = self.blocks[i](result, training, mask)

        return result


class Decoder(tf.keras.layers.Layer):
    """Create the decoder for the transformer"""

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


class Transformer(tf.keras.Model):
    """Transformer model class"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Transformer class initializer"""
        super().__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Calls the transformer into action"""
        enc = self.encoder(inputs, training, encoder_mask)
        dec = self.decoder(target, enc, training, look_ahead_mask,
                           decoder_mask)
        result = self.linear(dec)

        return result
