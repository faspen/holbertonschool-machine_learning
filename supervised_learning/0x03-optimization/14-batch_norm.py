#!/usr/bin/env python3
"""Batch upgrade"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return tensor of output"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(units=n, kernel_initializer=kernel)

    mean, var = tf.nn.moments(base(prev), axes=0)
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta')
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma')
    epsilon = 1e-08

    batch = tf.nn.batch_normalization(
        x=base(prev),
        mean=mean,
        variance=var,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)
    if activation is True:
        return activation(batch)
    return batch
