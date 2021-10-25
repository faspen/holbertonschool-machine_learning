#!/usr/bin/env python3
"""Correlation module"""


import numpy as np


def correlation(C):
    """Calculate correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    v = np.sqrt(np.diag(C))
    outer = np.outer(v, v)
    corr = C / outer
    return corr
