#!/usr/bin/env python3
"""Maximization module"""


import numpy as np


def maximization(X, g):
    """Calculate maximization step in EM algo for GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape

    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n,)).all():
        return None, None, None

    return pi, m, S
