#!/usr/bin/env python3
"""Likelihood module"""


import numpy as np


def likelihood(x, n, P):
    """Calculate likelihood of obtaining certain data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for counter in P:
        if counter > 1 or counter < 0:
            raise ValueError("All values in P must be in the range [0, 1]")

    fact = np.math.factorial
    fact_coef = fact(n) / (fact(n - x) * fact(x))
    likelihood = fact_coef * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
