#!/usr/bin/env python3
"""Bidirectional RNN"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for RNN"""
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t, m, h * 2))
    h_next = np.zeros((t, m, h))
    h_prev = np.zeros((t, m, h))
    Y = []
    x_f = h_0
    x_b = h_t

    for step in range(t):
        h_next[step] = bi_cell.forward(x_f, X[step])
        h_prev[t - step - 1] = bi_cell.backward(x_b, X[t - step - 1])
        x_b = h_prev[t - step - 1]
        x_f = h_next[step]

    H = np.concatenate((h_next, h_prev), axis=-1)
    Y = bi_cell.output(H)

    return H, np.array(Y)
