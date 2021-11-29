#!/usr/bin/env python3
"""Recurrent Neural Network"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward prop for simple RNN"""
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    for i in range(t):
        h_next, y = rnn_cell.forward(H[i], X[i])
        H[i + 1] = h_next
        if i == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    sh = Y.shape[-1]
    Y = Y.reshape(t, m, sh)
    return H, Y
