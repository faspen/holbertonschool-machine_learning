#!/usr/bin/env python3
"""One hot decode"""


import numpy as np


def one_hot_decode(one_hot):
    """Decode matrix to vector"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    vector = one_hot.argmax(0)

    return vector
