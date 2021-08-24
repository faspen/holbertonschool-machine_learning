#!/usr/bin/env python3
"""L2 Regularization layer"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Return output of the new layer"""
    function = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regular = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        units=n,
        name='layer',
        activation=activation,
        kernel_initializer=function,
        kernel_regularizer=regular)
    return layer(prev)
