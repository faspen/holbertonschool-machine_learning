#!/usr/bin/env python3
"""Train the model"""


import tensorflow.keras as K


def save_model(network, filename):
    """Save the model"""
    network.save(filename)
    return None


def load_model(filename):
    """Load the model"""
    return K.models.load_model(filename)
