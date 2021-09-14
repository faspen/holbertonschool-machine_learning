#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Return densenet keras model"""
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    fil = 2 * growth_rate

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=fil, kernel_size=7,
                            padding='same', strides=2,
                            kernel_initializer=init)(act1)

    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                     padding='same')(conv1)

    d_block1, fil = dense_block(max_pool, fil, growth_rate, 6)
    t_layer1, fil = transition_layer(d_block1, fil, compression)

    d_block2, fil = dense_block(t_layer1, fil, growth_rate, 12)
    t_layer2, fil = transition_layer(d_block2, fil, compression)

    d_block3, fil = dense_block(t_layer2, fil, growth_rate, 24)
    t_layer3, fil = transition_layer(d_block3, fil, compression)

    d_block4, fil = dense_block(t_layer3, fil, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding='valid')(d_block4)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)
    model = K.Model(inputs=X, outputs=dense)

    return model
