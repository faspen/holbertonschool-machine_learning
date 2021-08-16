#!/usr/bin/env python3
"""Shuffle data"""


import numpy as np


def shuffle_data(X, Y):
    """Return shuffled matrices"""
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]
