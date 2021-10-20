#!/usr/bin/env python3
"""Definiteness calculator"""


import numpy as np


def definiteness(matrix):
    """Return whether or not matrix is definite"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.linalg.eig(matrix):
        return None
    if not(matrix.transpose() == matrix).all():
        return None

    alp, v = np.linalg.eig(matrix)

    if np.all(alp == 0):
        return None
    if np.all(alp > 0):
        return "Positive definite"
    if np.all(alp >= 0):
        return "Positive semi-definite"
    if np.all(alp < 0):
        return "Negative definite"
    if np.all(alp <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
