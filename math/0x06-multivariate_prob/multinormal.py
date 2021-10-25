#!/usr/bin/env python3
"""Initialize module"""


import numpy as np


class MultiNormal:
    """Multivariate Normal distribution class"""

    def __init__(self, data):
        """Class constructor for multinormal"""

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean_cov(data)

    def mean_cov(self, X):
        """Calculate the mean and covariance of data"""
        d = X.shape[0]
        n = X.shape[1]
        self.mean = np.mean(X, axis=1).reshape(d, 1)
        X -= self.mean
        self.cov = ((np.dot(X, X.T)) / (n - 1))

    def pdf(self, x):
        """Calculate the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if (len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_mean = x - self.mean
        pdf = (1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov)))
               * np.exp(-(np.linalg.solve(self.cov, x_mean).T.dot(x_mean))
               / 2))

        return float(pdf)
