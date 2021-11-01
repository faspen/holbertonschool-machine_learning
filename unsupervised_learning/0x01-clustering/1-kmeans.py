#!/usr/bin/env python3
"""K-means module"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on dataset"""
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    centroids = np.random.uniform(
        np.min(X, axis=0), np.max(X, axis=0), size=(k, d))

    for i in range(iterations):
        tmp = centroids.copy()
        M = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(M, axis=0)

        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = (X[clss == j]).mean(axis=0)
        M = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(M, axis=0)
        if np.all(tmp == centroids):
            return centroids, clss

    return centroids, clss
