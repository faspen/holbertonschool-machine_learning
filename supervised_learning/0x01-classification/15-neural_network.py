#!/usr/bin/env python3
"""Neural Network"""


import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate gradient descent"""
        dz1 = A2 - Y
        dw1 = np.matmul(A1, dz1.T) / A1.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / A2.shape[1]

        da = A1 * (1 - A1)
        dz2 = np.matmul(self.__W2.T, dz1)
        dz2 *= da
        dw2 = np.matmul(X, dz2.T) / A1.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / A1.shape[1]

        self.__W2 = self.__W2 - alpha * dw1.T
        self.__b2 = self.__b2 - alpha * db1
        self.__W1 = self.__W1 - alpha * dw2.T
        self.__b1 = self.__b1 - alpha * db2

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
        if graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(iterations + 1):
            prog1, prog2 = self.forward_prop(X)
            self.gradient_descent(X, Y, prog1, prog2, alpha)
            cost.append(self.cost(Y, self.__A2))

            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost[i]))
        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost)
            plt.title("Training cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)
