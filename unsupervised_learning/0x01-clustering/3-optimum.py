#!/usr/bin/env python3
"""Optimize k module"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for optimum number of clusters"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = []
    d_vars = []

    for i in range(kmin, kmax + 1):
        M, clss = kmeans(X, i, iterations)
        results.append((M, clss))

        if i == kmin:
            first = variance(X, M)
        var = variance(X, M)
        d_vars.append(first - var)

    return results, d_vars
