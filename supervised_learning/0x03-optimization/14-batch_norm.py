#!/usr/bin/env python3
"""Batch upgrade"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return tensor of output"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(units=n, kernel_initializer=kernel)
    A = base(prev)
    mean, var = tf.nn.moments(A, axes=[0])
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)

    batch = tf.nn.batch_normalization(A, mean, var, beta, gamma, 1e-8)
    return activation(batch)
