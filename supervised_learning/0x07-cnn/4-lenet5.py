#!/usr/bin/env python3
"""LeNet-5 (Tensorflow)"""


import tensorflow as tf


def lenet5(x, y):
    """Build lenet5 in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer()

    conv_1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                              padding='same', activation=tf.nn.relu,
                              kernel_initializer=init)(x)
    pool_1 = tf.layers.MaxPooling2D(pool_size=2, padding='valid',
                                    strides=2)(conv_1)

    conv_2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                              padding='valid', activation=tf.nn.relu,
                              kernel_initializer=init)(pool_1)
    pool_2 = tf.layers.MaxPooling2D(pool_size=2, padding='valid',
                                    strides=2)(conv_2)

    flat = tf.layers.Flatten()(pool_2)

    full_1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=init)(flat)
    full_2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=init)(full_1)

    soft_out = tf.layers.Dense(units=10, kernel_initializer=init)(full_2)

    output = tf.nn.softmax(soft_out)
    loss = tf.losses.softmax_cross_entropy(y, soft_out)
    train = tf.train.AdamOptimizer().minimize(loss)

    v = tf.argmax(y, 1)
    pred = tf.argmax(soft_out, 1)
    eq = tf.equal(pred, v)
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))

    return output, train, loss, accuracy
