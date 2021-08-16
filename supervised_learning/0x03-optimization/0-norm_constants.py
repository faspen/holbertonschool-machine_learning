#!/usr/bin/env python3
"""Normalize constansts"""


import numpy as np


def normalization_constants(X):
    """Return mean and stddev of each feature"""
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
