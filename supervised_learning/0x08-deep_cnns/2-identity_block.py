#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Return identity block"""
    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=None)

    conv_1 = K.layers.Conv2D(filters=F11, kernel_size=1,
                             padding='same',
                             kernel_initializer=init)(A_prev)

    batch_norm_1 = K.layers.BatchNormalization()(conv_1)
    activation_1 = K.layers.Activation('relu')(batch_norm_1)

    conv_2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                             padding='same',
                             kernel_initializer=init)(activation_1)

    batch_norm_2 = K.layers.BatchNormalization()(conv_2)
    activation_2 = K.layers.Activation('relu')(batch_norm_2)

    conv_3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                             padding='same',
                             kernel_initializer=init)(activation_2)

    batch_norm_3 = K.layers.BatchNormalization()(conv_3)
    add_layer = K.layers.Add()([batch_norm_3, A_prev])
    activation_3 = K.layers.Activation('relu')(add_layer)

    return activation_3
