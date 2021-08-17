#!/usr/bin/env python3
"""RMSProp upgrade"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Return optimization operation"""
    return tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)
