#!/usr/bin/env python3
"""Deep neural network module"""


import numpy as np


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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for index, section in enumerate(layers):
            if not isinstance(section, int) or section < 0:
                raise TypeError("layers must be a list of positive integers")
            if index is 0:
                init = np.random.randn(section, nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(index + 1)] = init
            if index > 0:
                dv1 = np.random.randn(section, layers[index - 1])
                dv2 = np.sqrt(2 / layers[index - 1])
                self.__weights['W' + str(index + 1)] = dv1 * dv2
            self.__weights['b' + str(index + 1)] = np.zeros((section, 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Get the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def sigmoid(self, v):
        """Sigmoid function"""
        return (1 / (1 + np.exp(-v)))

    def forward_prop(self, X):
        """Forward propagation"""
        for index in range(self.__L + 1):
            if index is 0:
                self.__cache['A' + str(index)] = X
            else:
                weighting = self.__weights['W' + str(index)]
                biasing = self.__weights['b' + str(index)]
                store = self.__cache['A' + str(index - 1)]
                v = (np.matmul(weighting, store)) + biasing
                self.__cache['A' + str(index)] = self.sigmoid(v)
        return (self.__cache['A' + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """Calculate cost"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        chest, self.__cache = self.forward_prop(X)
        predict = np.where(chest >= 0.5, 1, 0)
        cost = self.cost(Y, chest)
        return predict, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""
        m = Y.shape[1]
        weights = self.weights.copy()

        for i in range(self.L, 0, -1):
            if i != self.L:
                dzi = np.multiply(np.matmul(
                    weights['W' + str(i + 1)].T, dzi),
                    (self.cache['A' + str(i)] *
                     (1 - self.cache['A' + str(i)])))
            else:
                dzi = self.cache['A' + str(i)] - Y
            dwi = (np.matmul(dzi, self.cache['A' + str(i - 1)].T) / m)
            dbi = (np.sum(dzi, axis=1, keepdims=True) / m)

            self.__weights['W' + str(i)] = weights['W' + str(i)] - alpha * dwi
            self.__weights['b' + str(i)] = weights['b' + str(i)] - alpha * dbi
