#!/usr/bin/env python3
"""Test model"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Return loss and acc of model with testing data"""
    return network.evaluate(data, labels, verbose=verbose)
