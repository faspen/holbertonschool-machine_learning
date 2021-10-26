#!/usr/bin/env python3
"""Initialize t-SNE"""


import numpy as np


def P_init(X, perplexity):
    """Initialize all variables in t-SNE"""
    n, d = X.shape
    X_sum = np.sum(np.square(X), 1)

    D = np.add(np.add(-2 * np.dot(X, X.T), X_sum).T, X_sum)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
