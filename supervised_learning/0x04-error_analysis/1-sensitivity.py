#!/usr/bin/env python3
"""Sensitivity module"""


import numpy as np


def sensitivity(confusion):
    """Return array shape (classes,)"""
    pos = np.sum(confusion, axis=1)
    true = np.diagonal(confusion)
    res = true / pos
    return res
