#!/usr/bin/env python3
"""Neural style transfer"""


import numpy as np
import tensorflow as tf


class NST():
    """Class the performs neural style transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class initializer"""
        if not isinstance(
                style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if isinstance(
                content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        tf.enable_eager_execution()

        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales image appropriately"""
        tf.enable_eager_execution()

        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        height, width, _ = image.shape

        if height > width:
            new_h = 512
            new_w = (new_h * width) / height
        else:
            new_w = 512
            new_h = (new_w * height) / width

        resized = tf.image.resize_images(
            image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        return resized
