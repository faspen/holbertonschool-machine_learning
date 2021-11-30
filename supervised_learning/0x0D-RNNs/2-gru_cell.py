#!/usr/bin/env python3
"""GRU Cell"""


import numpy as np


class GRUCell():
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """GRUCell class constructor"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation"""
        concat1 = np.concatenate((h_prev, x_t), axis=1)
        r = np.matmul(concat1, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        z = np.matmul(concat1, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        concat2 = np.concatenate((r * h_prev, x_t), axis=1)
        h_tmp = np.matmul(concat2, self.Wh) + self.bh
        h_tmp = np.tanh(h_tmp)
        h_next = (1 - z) * h_prev + z * h_tmp

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
