#!/usr/bin/env python3
"""Batch upgrade"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return tensor of output"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(units=n, name='layer', kernel_initializer=kernel)
    A = base(prev)
    mean, var = tf.nn.moments(A, [0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        trainable=True)

    batch = tf.nn.batch_normalization(
        A,
        mean,
        var,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8)
    if activation is True:
        return activation(batch)
    return batch
