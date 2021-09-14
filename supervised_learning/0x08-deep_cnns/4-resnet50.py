#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Return resnet keras model"""
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', kernel_initializer=init)(X)
    batch_norm_1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(batch_norm_1)
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(act1)

    conv2_x1 = projection_block(max_pool, [64, 64, 256], 1)
    conv2_x2 = identity_block(conv2_x1, [64, 64, 256])
    conv2_x3 = identity_block(conv2_x2, [64, 64, 256])

    conv3_x1 = projection_block(conv2_x3, [128, 128, 512])
    conv3_x2 = identity_block(conv3_x1, [128, 128, 512])
    conv3_x3 = identity_block(conv3_x2, [128, 128, 512])
    conv3_x4 = identity_block(conv3_x3, [128, 128, 512])

    conv4_x1 = projection_block(conv3_x4, [256, 256, 1024])
    conv4_x2 = identity_block(conv4_x1, [256, 256, 1024])
    conv4_x3 = identity_block(conv4_x2, [256, 256, 1024])
    conv4_x4 = identity_block(conv4_x3, [256, 256, 1024])
    conv4_x5 = identity_block(conv4_x4, [256, 256, 1024])
    conv4_x6 = identity_block(conv4_x5, [256, 256, 1024])

    conv5_x1 = projection_block(conv4_x6, [512, 512, 2048])
    conv5_x2 = identity_block(conv5_x1, [512, 512, 2048])
    conv5_x3 = identity_block(conv5_x2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(conv5_x3)

    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)

    model = K.Model(inputs=X, outputs=dense)

    return model
