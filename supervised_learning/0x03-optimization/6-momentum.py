#!/usr/bin/env python3
"""Momentum upgrade"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Return momentum operation"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
