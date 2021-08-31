#!/usr/bin/env python3
"""Predict model"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Return data prediction"""
    return network.predict(data, verbose=verbose)
