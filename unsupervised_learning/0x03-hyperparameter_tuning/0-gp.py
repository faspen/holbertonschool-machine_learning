#!/usr/bin/env python3
"""Initialize Gaussian Process"""


import numpy as np


class GaussianProcess():
    """Noiseless 1D Gaussian Process class"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Constructs the class"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix"""
        shaper = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        sqdist = shaper + second - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
