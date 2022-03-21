#!/usr/bin/env python3
"""Crop task"""


import tensorflow as tf


def crop_image(image, size):
    """Crop an image"""
    return tf.image.random_crop(image, size=size)