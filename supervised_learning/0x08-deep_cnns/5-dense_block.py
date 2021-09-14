#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Return concat of each layer"""
    init = K.initializers.he_normal()

    for i in range(layers):
        batch_norm_1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(batch_norm_1)
        bottle = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=(1, 1),
                                 padding='same', strides=(1, 1),
                                 kernel_initializer=init)(act1)

        batch_norm_2 = K.layers.BatchNormalization()(bottle)
        act2 = K.layers.Activation('relu')(batch_norm_2)
        conv = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                               padding='same', strides=(1, 1),
                               kernel_initializer=init)(act2)

        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
