#!/usr/bin/env python3
"""Convolutional forward prop"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Forward prop over conv layer of nn"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding is 'same':
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pw = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    if padding is 'valid':
        ph, pw = 0, 0

    A_prev = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw),
                    (0, 0)], 'constant', constant_values=0)
    zh = ((h_prev + (2 * ph) - kh) // sh) + 1
    zw = ((w_prev + (2 * pw) - kw) // sw) + 1
    conv = np.zeros((m, zh, zw, c_new))
    i = np.arange(0, m)

    for z in range(c_new):
        kern = W[:, :, :, z]
        i = 0
        for h in range(zh):
            j = 0
            for w in range(zw):
                output = np.sum(
                    A_prev[:, h:h + kh, w:w + kw, :] * kern,
                    axis=1).sum(axis=1).sum(axis=1)
                output += b[0, 0, 0, z]
                conv[:, i, j, z] = activation(output)
                j += 1
            i += 1
    return conv

    return conv
