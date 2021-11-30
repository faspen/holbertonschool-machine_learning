#!/usr/bin/env python3
"""LSTM Cell"""


import numpy as np


class LSTMCell():
    """Class the represents LSTM unit"""

    def __init__(self, i, h, o):
        """LSTMCell class constructor"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid helper function"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation"""
        concat1 = np.concatenate((h_prev, x_t), axis=1)
        rf = np.matmul(concat1, self.Wf) + self.bf
        rf = self.sigmoid(rf)
        ru = np.matmul(concat1, self.Wu) + self.bu
        ru = self.sigmoid(ru)
        rc = np.matmul(concat1, self.Wc) + self.bc
        rc = np.tanh(rc)
        ro = np.matmul(concat1, self.Wo) + self.bo
        ro = self.sigmoid(ro)

        c_next = (ru * rc) + (rf * c_prev)
        h_next = ro * np.tanh(c_next)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
