#!/usr/bin/env python3
"""Create masks for model"""


import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Create masks for training and validation"""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    sizer = target.shape[1]
    look = 1 - tf.linalg.band_part(tf.ones((sizer, sizer)), -1, 0)
    tmp_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    tmp_mask = tmp_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(tmp_mask, look)

    return encoder_mask, combined_mask, decoder_mask
