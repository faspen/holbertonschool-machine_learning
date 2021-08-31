#!/usr/bin/env python3
"""Save and Load Configuration"""


import tensorflow.keras as K


def save_config(network, filename):
    """Save model in JSON"""
    json_model = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_model)
    return None


def load_config(filename):
    """Load from JSON"""
    with open(filename, 'r') as f:
        json_read = f.read()
    return K.models.model_from_json(json_read)
