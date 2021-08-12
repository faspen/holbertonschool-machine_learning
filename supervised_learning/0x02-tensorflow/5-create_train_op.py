#!/usr/bin/env python3
"""Training operation"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Create training routine for network"""
    operation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return operation
