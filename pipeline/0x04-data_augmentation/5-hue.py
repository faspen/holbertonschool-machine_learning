#!/usr/bin/env python3
"""Hue task"""


import tensorflow as tf


def change_hue(image, delta):
    """Change the hue"""
    return tf.image.adjust_hue(image, delta)
