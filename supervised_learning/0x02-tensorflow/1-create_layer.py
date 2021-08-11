#!/usr/bin/env python3
"""Create a layer"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that creates a layer"""
    function = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(units=n,
                            name='layer',
                            activation=activation,
                            kernel_initializer=function)
    return layer(prev)
