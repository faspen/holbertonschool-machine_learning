#!/usr/bin/env python3
"""Bidirectional Output"""


import numpy as np


class BidirectionalCell():
    """Represents bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Class constructor for Bidirectional Cell"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate hidden state in forward direction"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward direction"""
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.matmul(concat, self.Whb) + self.bhb
        h_prev = np.tanh(h_prev)

        return h_prev

    def output(self, H):
        """Calculate all outputs for the RNN"""
        t, m, h = H.shape
        s = self.by.shape[-1]
        Y = np.zeros((t, m, s))

        for steps in range(t):
            Y[steps] = np.matmul(H[steps], self.Wy) + self.by
            Y[steps] = np.exp(Y[steps]) / np.sum(np.exp(Y[steps]),
                                                 axis=1, keepdims=True)

        return Y
