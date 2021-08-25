#!/usr/bin/env python3
"""Create layer with dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Return output of new layer using tensorflow"""
    dropout = tf.layers.Dropout(keep_prob)
    function = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=function,
        kernel_regularizer=dropout)
    return layer(prev)
