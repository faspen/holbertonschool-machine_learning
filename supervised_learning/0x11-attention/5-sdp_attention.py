#!/usr/bin/env python3
"""Scaled dot product attention"""


import tensorflow as tf


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
