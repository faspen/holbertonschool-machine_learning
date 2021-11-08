#!/usr/bin/env python3
"""Regular chains"""


import numpy as np


def regular(P):
    """Determine steady state probability of chain"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, n_s = P.shape

    if n != n_s:
        return None
    sums = np.sum(P, axis=1)

    for i in sums:
        if not np.isclose(i, 1):
            return None

    evals, enums = np.linalg.eig(P.T)
    new = enums / enums.sum()
    new = new.real

    for j in np.dot(new.T, P):
        if (j >= 0).all() and np.isclose(j.sum(), 1):
            return j.reshape(1, n)

    return None
