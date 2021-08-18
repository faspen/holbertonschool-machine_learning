#!/usr/bin/env python3
"""Batch normalization"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Return normalized Z matrix"""
    av = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    norm = (Z - av) / np.sqrt(var + epsilon)
    new_arr = gamma * norm + beta
    return new_arr
