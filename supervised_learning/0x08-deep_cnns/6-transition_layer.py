#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Output of trans layer, number of filters"""
    init = K.initializers.he_normal()
    fil = int(nb_filters * compression)

    batch_norm = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=fil, kernel_size=1,
                           padding='same', strides=1,
                           kernel_initializer=init)(act)

    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same')(conv)

    return avg_pool, fil
