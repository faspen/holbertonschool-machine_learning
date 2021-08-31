#!/usr/bin/env python3
"""Save and Load Weights"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Save weights and return none"""
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load the weights and return none"""
    network.load_weights(filename)
    return None
