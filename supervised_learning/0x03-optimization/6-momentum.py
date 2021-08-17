#!/usr/bin/env python3
"""Momentum upgrade"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Return momentum operation"""
    tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)
