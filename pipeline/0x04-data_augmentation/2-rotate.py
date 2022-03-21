#!/usr/bin/env python3
"""Rotate task"""


import tensorflow as tf


def rotate_image(image):
    """Rotate an image 90 degrees"""
    return tf.image.rot90(image)
