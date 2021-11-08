#!/usr/bin/env python3
"""Markov chain"""


import numpy as np


def markov_chain(P, s, t=1):
    """Predict probability of being in specific state"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, n_s = P.shape

    if n != n_s:
        return None
    if not isinstance(s, np.ndarray):
        return None
    if s.ndim != 2 or s.shape[0] != 1 or s.shape[1] != n:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    sums = np.sum(P, axis=1)
    for i in sums:
        if not np.isclose(i, 1):
            return None

    tmp = s
    tz = np.zeros((1, n))
    for i in range(t):
        tz = np.matmul(tmp, P)
        tmp = tz

    return tz
