#!/usr/bin/env python3
"""Convolution with channels"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Return the convolved images"""
    m, h_image, w_image, dummy = images.shape
    k_height = kernels.shape[0]
    k_width = kernels.shape[1]
    nc = kernels.shape[3]
    h_stride, w_stride = stride

    if padding == 'same':
        h_padding = ((h_image - 1) * h_stride + k_height - h_image) // 2 + 1
        w_padding = ((w_image - 1) * w_stride + k_width - w_image) // 2 + 1
    elif padding == 'valid':
        h_padding, w_padding = (0, 0)
    else:
        h_padding, w_padding = padding

    i_height = (images.shape[1] + 2 * h_padding - k_height) // h_stride + 1
    i_width = (images.shape[2] + 2 * w_padding - k_width) // w_stride + 1

    img_pad = np.pad(images, [(0, 0), (h_padding, h_padding),
                     (w_padding, w_padding), (0, 0)], mode='constant',
                     constant_values=0)
    result = np.zeros((m, i_height, i_width, nc))

    for i in range(i_height):
        for j in range(i_width):
            for k in range(nc):
                padder = img_pad[:, i * h_stride:i * h_stride + k_height,
                                 j * w_stride:j * w_stride + k_width]
                arr_sum = np.sum(padder * kernels[:, :, :, k], axis=(1, 2, 3))
                result[:, i, j, k] = arr_sum
    return result
