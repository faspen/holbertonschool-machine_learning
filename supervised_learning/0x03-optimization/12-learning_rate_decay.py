#!/usr/bin/env python3
"""Learning rate decay upgrade"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return decay rate operation"""
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True)
