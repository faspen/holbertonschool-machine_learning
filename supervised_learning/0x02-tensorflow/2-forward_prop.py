#!/usr/bin/env python3
"""Forward propagation"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward prop function"""
    layer = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
