#!/usr/bin/env python3
"""Optimize keras"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Return None"""
    opt = K.optimizers.Adam(alpha, beta_1=beta1, beta_2=beta2)

    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return None
