#!/usr/bin/env python3
"""Convolutional forward prop"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Forward prop over conv layer of nn"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding is 'same':
        ph = int(((sh * h_prev) - sh + kh - h_prev) // 2)
        pw = int(((sw * w_prev) - sw + kw - w_prev) // 2)
    if padding is 'valid':
        ph = 0
        pw = 0

    dA_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw),
                              (0, 0)), 'constant', constant_values=0)
    zh = int((h_prev + (2 * ph) - kh) // sh) + 1
    zw = int((w_prev + (2 * pw) - kw) // sw) + 1
    output = np.zeros((m, zh, zw, c_new))
    img = np.arange(0, m)

    for h in range(zh):
        for w in range(zw):
            for z in range(c_new):
                output[img, h, w, z] = activation(
                    (np.sum(np.multiply(
                        dA_prev[img, h * sh:kh + h * sh, w * sw:kw + w * sw],
                        W[:, :, :, z]),
                        axis=(1, 2, 3))) + b[0, 0, 0, z])

    return output
