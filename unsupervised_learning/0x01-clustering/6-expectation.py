#!/usr/bin/env python3
"""Expectation module"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step in EM algo for GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape

    if pi.shape[0] > n:
        return None, None
    i = pi.shape[0]
    if m.shape[0] != i or m.shape[1] != d:
        return None, None
    if S.shape[0] != i or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    z = np.zeros((i, n))

    for j in range(i):
        PDF = pdf(X, m[j], S[j])
        z[j] = pi[j] * PDF

    sums = np.sum(z, axis=0, keepdims=True)
    z /= sums

    logs = np.sum(np.log(sums))

    return z, logs
