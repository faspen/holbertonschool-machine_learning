#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    conv_1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)

    conv_2 = K.layers.Conv2D(filters=F3R, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)

    conv_3 = K.layers.Conv2D(filters=F3, kernel_size=3,
                             padding='same', activation='relu',
                             kernel_initializer=init)(conv_2)

    conv_4 = K.layers.Conv2D(filters=F5R, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)

    pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=1,
                                 padding='same')(A_prev)

    conv_5 = K.layers.Conv2D(filters=F5, kernel_size=5,
                             padding='same', activation='relu',
                             kernel_initializer=init)(conv_4)

    conv_6 = K.layers.Conv2D(filters=FPP, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(pool)

    output = K.layers.concatenate([conv_1, conv_3, conv_5, conv_6])

    return output
