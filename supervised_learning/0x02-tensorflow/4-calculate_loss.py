#!/usr/bin/env python3
"""Calculate the loss"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate softmax loss"""
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN)
    return loss
