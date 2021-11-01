#!/usr/bin/env python3
"""Initialize K-means"""


import numpy as np


def initialize(X, k):
    """Initialize cluster centroids"""
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d = X.shape
    clusters = np.random.uniform(
        np.min(X, axis=0), np.max(X, axis=0), size=(k, d))

    return clusters
