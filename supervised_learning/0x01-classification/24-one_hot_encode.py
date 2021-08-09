#!/usr/bin/env python3
"""One hot encode"""


import numpy as np


def one_hot_encode(Y, classes):
    """Convert vector to one hot matrix"""
    if not isinstance(
            Y,
            np.ndarray) or not isinstance(
            classes,
            int) or Y.ndim != 1:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((m, classes))
        row = np.arange(m)
        one_hot[row, Y] = 1
    except Exception:
        return None

    return one_hot.T
