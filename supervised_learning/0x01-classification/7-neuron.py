#!/usr/bin/env python3
"""Neuron class"""


import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        self.__A = self.forward_prop(X)
        result = np.where(self.__A >= 0.5, 1, 0)
        return result, self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gd"""
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Train the model"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        cost = []

        if verbose is True and iterations % step == 0:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(iterations + 1):
            progress = self.forward_prop(X)
            self.gradient_descent(X, Y, progress, alpha)
            cost.append(self.cost(Y, self.__A))
            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost[i]))

        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost)
            plt.title("Training cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)
