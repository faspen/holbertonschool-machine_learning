#!/usr/bin/env python3
"""Bidirectional Cell Backward"""


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
