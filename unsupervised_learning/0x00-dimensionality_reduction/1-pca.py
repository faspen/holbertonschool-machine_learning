#!/usr/bin/env python3
"""PCA module for task 1"""


import numpy as np


def pca(X, ndim):
    """Perform PCA on dataset"""
    X_mean = X - X.mean(axis=0)
    s, v, d = np.linalg.svd(X_mean)

    W = d.T
    Wm = W[:, 0:ndim]
    T = np.matmul(X_mean, Wm)

    return T
