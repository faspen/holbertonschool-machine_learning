#!/usr/bin/env python3
"""Convolution with channels"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Return the convolved images"""
    m, h_image, w_image, dummy = images.shape
    k_height = kernel_shape[0]
    k_width = kernel_shape[1]
    h_stride, w_stride = stride

    i_height = (h_image - k_height) // h_stride + 1
    i_width = (w_image - k_width) // w_stride + 1

    result = np.zeros((m, i_height, i_width, dummy))

    for i in range(i_height):
        for j in range(i_width):
            padder = images[:, i * h_stride:i * h_stride + k_height,
                            j * w_stride:j * w_stride + k_width]
            if mode == 'max':
                result[:, i, j] = np.max(padder, axis=(1, 2))
            if mode == 'avg':
                result[:, i, j] = np.mean(padder, axis=(1, 2))
    return result
    return result
