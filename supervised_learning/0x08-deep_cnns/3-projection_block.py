#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Return projection block output"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1,
                            padding='same', strides=s,
                            kernel_initializer=init)(A_prev)
    batch_norm_1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(batch_norm_1)

    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                            padding='same', strides=1,
                            kernel_initializer=init)(act1)
    batch_norm_2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(batch_norm_2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                            padding='same', strides=1,
                            kernel_initializer=init)(act2)
    conv4 = K.layers.Conv2D(filters=F12, kernel_size=1,
                            padding='same', strides=s,
                            kernel_initializer=init)(A_prev)
    batch_norm_3 = K.layers.BatchNormalization()(conv3)
    batch_norm_4 = K.layers.BatchNormalization()(conv4)

    add = K.layers.Add()([batch_norm_3, batch_norm_4])
    act3 = K.layers.Activation('relu')(add)

    return act3
