#!/usr/bin/env python3
"""Intersection module"""


import numpy as np


def intersection(x, n, P, Pr):
    """Calculate intersection of obtaining certain data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for counter in range(P.shape[0]):
        if P[counter] > 1 or P[counter] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[counter] > 1 or Pr[counter] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    fact = np.math.factorial
    fact_coef = fact(n) / (fact(n - x) * fact(x))
    likelihood = fact_coef * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr
    return intersection
