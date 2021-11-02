#!/usr/bin/env python3
"""PDF module"""


import numpy as np


def pdf(X, m, S):
    """Calculate probability density function of Gaussian distribution"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = X.shape[0]

    A = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))
    B = np.matmul(np.linalg.inv(S), (X - m).T)
    C = np.expp(-0.5 * np.sum((X - m).T * M, axis=0))
    P = A * C
    P = np.maximum(P, 1e-300)

    return P
