#!/usr/bin/env python3
"""Neuron class"""


import numpy as np


class Neuron():
    """class for neuron"""

    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter W"""
        return self.__W

    @ property
    def b(self):
        """Getter b"""
        return self.__b

    @ property
    def A(self):
        """Getter A"""
        return self.__A

    def forward_prop(self, X):
        """Binary prop"""
        value = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(value)
        return self.__A

    def sigmoid(self, v):
        """Sigmoid helper"""
        return (1 / (1 + np.exp(-v)))

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * (np.log(1.0000001 - A)))
        return cost
