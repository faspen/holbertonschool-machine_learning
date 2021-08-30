#!/usr/bin/env python3
"""Input Keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Return keras model"""
    inputs = K.layers.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(
                layers[0],
                activation=activations[0],
                kernel_regularizer=reg)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg)(dropout)
    model = K.models.Model(inputs=inputs, outputs=output)
    return model
