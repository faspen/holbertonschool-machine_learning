#!/usr/bin/env python3
"""PCA module for task 0"""


import numpy as np


def pca(X, var=0.95):
    """Perform PCA on dataset"""
    s, v, d = np.linalg.svd(X)
    ratios = list(i / np.sum(v) for i in v)
    variant = np.cumsum(ratios)
    res = np.argwhere(variant >= var)[0, 0]
    W = d.T[:, :(res + 1)]
    return (W)
