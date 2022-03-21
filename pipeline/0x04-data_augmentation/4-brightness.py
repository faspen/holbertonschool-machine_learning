#!/usr/bin/env python3
"""Brightness task"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """Change the brightness"""
    return tf.image.adjust_brightness(image, max_delta)
