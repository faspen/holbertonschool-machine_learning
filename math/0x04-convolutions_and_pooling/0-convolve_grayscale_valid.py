#!/usr/bin/env python3
"""Valid Convolution"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Return array containing images"""
    m = images.shape[0]
    i_height = images.shape[1]
    i_width = images.shape[2]
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]

    h_output = i_height - k_height + 1
    w_output = i_width - k_width + 1

    result = np.zeros((m, h_output, w_output))

    im = np.arange(m)

    for i in range(h_output):
        for j in range(w_output):
            stride = images[im, i:k_height + i, j:k_width + j]
            result[im, i, j] = np.sum(np.multiply(stride, kernel), axis=(1, 2))
    return result
