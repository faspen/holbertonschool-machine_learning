#!/usr/bin/env python3
"""Neural Network"""


import numpy as np


class NeuralNetwork():
    """Neural network class"""

    def __init__(self, nx, nodes):
        """Constructor for nn"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """w1 getter"""
        return self.__W1

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W2(self):
        """w2 getter"""
        return self.__W2

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2

    def sigmoid(self, v):
        """Sigmoid helper"""
        return (1 / (1 + np.exp(-v)))

    def forward_prop(self, X):
        """Forward propagation"""
        value = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(value)
        value2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(value2)
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculate cost"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        self.__A = self.forward_prop(X)
        result = np.where(self.__A2 >= 0.5, 1, 0)
        return result, self.cost(Y, self.__A2)
