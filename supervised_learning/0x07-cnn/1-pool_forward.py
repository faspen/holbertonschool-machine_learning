#!/usr/bin/env python3
"""Pooling Forward Prop"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Forward Prop over pooling layer of nn"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    zh = int(((h_prev - kh) // sh) + 1)
    zw = int(((w_prev - kw) // sw) + 1)

    conv = np.zeros((m, zh, zw, c_prev))
    i = np.arange(m)

    for h in range(zh):
        for w in range(zw):
            if mode is 'max':
                conv[i, h, w] = np.max(A_prev[i, h * sh:kh + (h * sh),
                                       w * sw:kw + (w * sw)], axis=(1, 2))
            if mode is 'mean':
                conv[i, h, w] = np.mean(A_prev[i, h * sh:kh + (h * sh),
                                        w * sw:kw + (w * sw)], axis=(1, 2))
    return conv
