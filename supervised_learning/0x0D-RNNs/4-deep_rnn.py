#!/usr/bin/env python3
"""Deep RNN"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation on deep RNN"""
    h_next = h_0
    L, m, i = h_0.shape
    H = np.zeros((X.shape[0] + 1, L, m, i))
    H[0] = h_0
    Y = []
    for step in range(X.shape[0]):
        h_prev = X[step]
        for layer in range(L):
            h_next, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            h_prev = h_next
            H[step + 1, L, :, :] = h_next
        Y.append(y)
    return H, Y
