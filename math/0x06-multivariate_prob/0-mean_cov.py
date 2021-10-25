#!/usr/bin/env python3
"""Mean and Covariance"""


import numpy as np


def mean_cov(X):
    """Calculate mean and covariance of data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    X -= mean
    cov = ((np.dot(X.T, X)) / (n - 1))

    return mean, cov
