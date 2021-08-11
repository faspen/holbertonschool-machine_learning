#!/usr/bin/env python3
"""Calculate accuracy"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """validate predictions"""
    validate = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(validate, tf.float32))
    return acc
