#!/usr/bin/env python3
"""Precision module"""


import numpy as np


def precision(confusion):
    """Return array shape (classes,)"""
    relevant = np.sum(confusion, axis=0)
    retrieved = np.diagonal(confusion)
    res = retrieved / relevant
    return res
