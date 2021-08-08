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

        A1 = self.__cache['A' + str(self.__L)]
        A2 = self.__cache['A' + str(self.__L - 1)]
        W = self.__weights['W' + str(self.__L)]
        b = self.__weights['b' + str(self.__L)]
        dz_Diction = {}
        dz1 = A1 - Y

        dz_Diction['dz' + str(self.__L)] = dz1
        dw = (1 / m) * np.matmul(A2, dz1.T)
        db = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__weights['W' + str(self.__L)] = (W - alpha * dw.T)
        self.__weights['b' + str(self.__L)] = (b - alpha * db)

        for i in range(self.__L - 1, 0, -1):
            A_prog = self.__cache['A' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            W_prog = self.__weights['W' + str(i)]
            W_jump = self.__weights['W' + str(i + 1)]
            b_prog = self.__weights['b' + str(i)]

            dz2 = np.matmul(W_jump.T, dz_Diction['dz' + str(i + 1)])
            dz3 = A_prog * (1 - A_prog)
            dz32 = dz3 * dz2
            dw2 = ((1 / m) * np.matmul(A_prev, dz32.T))
            db2 = ((1 / m) * np.sum(dz32, axis=1, keepdims=True))
            dz_Diction['dz' + str(i)] = dz32

            self.__weights['W' + str(i)] = (W_prog - alpha * dw2.T)
            self.__weights['b' + str(i)] = (b_prog - alpha * db2)
