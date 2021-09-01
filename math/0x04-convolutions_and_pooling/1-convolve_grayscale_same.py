#!/usr/bin/env python3
"""Same Convolution"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Return the convoled images"""
    m = images.shape[0]
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]

    if k_height % 2 != 0:
        h_pad = (k_height - 1) // 2
    else:
        h_pad = k_height // 2
    if k_width % 2 != 0:
        w_pad = (k_width - 1) // 2
    else:
        w_pad = k_width // 2

    i_height = images.shape[1]
    i_width = images.shape[2]

    img_pad = np.pad(images, [(0, 0), (h_pad, h_pad), (w_pad, w_pad)],
                     mode='constant', constant_values=0)
    result = np.zeros((m, i_height, i_width))

    for i in range(i_height):
        for j in range(i_width):
            stride = img_pad[:, i:i + k_height, j:j + k_width] * kernel
            arr_sum = np.sum(stride, axis=(1, 2))
            result[:, i, j] = arr_sum
    return result
