#!/usr/bin/env python3
"""Initialize GMM"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a gaussian mixture model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape
    centroids, clss = kmeans(X, k, iterations=1000)
    pi = np.ones(k) / k
    m = centroids
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S
