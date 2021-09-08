#!/usr/bin/env python3
"""Pooling back prop"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Back prop over pooling network"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conv = np.zeros((m, h_prev, w_prev, c))

    for a in range(m):
        for k in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    i = h * sh
                    j = w * sw
                    if mode is 'max':
                        val = A_prev[a, i:i + kh, j:j + kw, k]
                        tmp = np.where(val == np.max(val), 1, 0)
                    elif mode is 'avg':
                        tmp = np.ones((kh, kw))
                        tmp /= (kh * kw)
                    conv[a, i:i + kh, j:j + kw, k] += (tmp * dA[a, h, w, k])

    return conv
