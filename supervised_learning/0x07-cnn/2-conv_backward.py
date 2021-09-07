#!/usr/bin/env python3
"""Convolutional Back Prop"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform back prop over convolutional layer of nn
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    tmp = A_prev

    if padding is 'same':
        ph = (((sh * h_prev) - sh + kh - h_prev) // 2)
        pw = (((sw * w_prev) - sw + kw - w_prev) // 2)
    if padding is 'valid':
        ph, pw = 0, 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    tmp_pad = np.pad(tmp, [(0, 0), (ph, ph), (pw, pw),
                     (0, 0)], 'constant', constant_values=0)
    dW = np.zeros_like(W)
    d_tmp = np.zeros(tmp_pad.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    d_tmp[i, h * sh:h * sh + kh, w * sw:w * sw +
                          kw, :] += dZ[i, h, w, c] * W[:, :, :, c]
                    dW[:, :, :, c] += tmp_pad[i, h * sh:h * sh +
                                              kh, w * sw:w * sw + kw, :] * \
                        dZ[i, h, w, c]
    if padding is 'same':
        d_tmp = d_tmp[:, ph:-ph, pw:-pw, :]
    else:
        d_tmp = d_tmp

    return d_tmp, dW, db
