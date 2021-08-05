#!/usr/bin/env python3
"""Deep neural network module"""


import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork():
    """DNN class"""

    def __init__(self, nx, layers):
        """Init for DNN"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for index, section in enumerate(layers):
            if not isinstance(section, int) or section < 0:
                raise TypeError("layers must be a list of positive integers")
            if index is 0:
                init = np.random.randn(section, nx) * np.sqrt(2 / nx)
                self.weights['W' + str(index + 1)] = init
            if index > 0:
                dv1 = np.random.randn(section, layers[index - 1])
                dv2 = np.sqrt(2 / layers[index - 1])
                self.weights['W' + str(index + 1)] = dv1 * dv2
            self.weights['b' + str(index + 1)] = np.zeros((section, 1))
