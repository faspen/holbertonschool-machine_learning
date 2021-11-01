#!/usr/bin/env python3
"""Variance module"""


import numpy as np


def variance(X, C):
    """Calculate total intra-cluster variance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    k, d = C.shape
    if not isinstance(k, int) or k <= 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    M = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    cluster = np.min(M, axis=0)
    var = np.sum(np.square(cluster))

    return var
