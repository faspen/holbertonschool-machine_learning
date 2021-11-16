#!/usr/bin/env python3
"""Gaussian Process Prediction"""


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

    def predict(self, X_s):
        """Predict mean and stdv of points in a GP"""
        K_1 = self.kernel(self.X, X_s)
        K_2 = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_1.T.dot(K_inv).dot(self.Y).reshape(X_s.shape[0])
        cov = K_2 - K_1.T.dot(K_inv).dot(K_1)
        sigma = np.diagonal(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update the Gaussian process"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
