#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Keras model for network"""
    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                             padding='same', activation='relu',
                             strides=(2, 2), kernel_initializer=init)(inputs)

    max_pool_1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                       strides=(2, 2), padding='same')(conv_1)

    conv_2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                             activation='relu', strides=(1, 1),
                             kernel_initializer=init)(max_pool_1)
    conv_3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                             padding='same', activation='relu',
                             strides=(1, 1), kernel_initializer=init)(conv_2)

    max_pool_2 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                       padding='same')(conv_3)

    conc_1 = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    conc_2 = inception_block(conc_1, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                       padding='same')(conc_2)

    conc_3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    conc_4 = inception_block(conc_3, [160, 112, 224, 24, 64, 64])
    conc_5 = inception_block(conc_4, [128, 128, 256, 24, 64, 64])
    conc_6 = inception_block(conc_5, [112, 144, 288, 32, 64, 64])
    conc_7 = inception_block(conc_6, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                       padding='same')(conc_7)

    conc_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    conc_9 = inception_block(conc_8, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(conc_9)

    drop = K.layers.Dropout(rate=0.4)(avg_pool)
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=init)(drop)

    model = K.Model(inputs=inputs, outputs=dense)

    return model
