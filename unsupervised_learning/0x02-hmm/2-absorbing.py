#!/usr/bin/env python3
"""Absorbing chains"""


import numpy as np


def absorbing(P):
    """Determine if markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False

    n, n_s = P.shape
    if n != n_s:
        return False
    sums = np.sum(P, axis=1)
    for i in sums:
        if not np.isclose(i, 1):
            return False

    diagonal = np.diag(P)
    if (diagonal == 1).all():
        return True

    absorb = (diagonal == 1)
    for x in range(len(diagonal)):
        for y in range(len(diagonal)):
            if P[x, y] > 0 and absorb[y]:
                absorb[x] = 1
    if (absorb == 1).all():
        return True

    return False
