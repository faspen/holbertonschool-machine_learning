#!/usr/bin/env python3
"""Sigma"""


def summation_i_squared(n):
    """sum total"""
    if n <= 0 or n is None:
        return None
    else:
        return sum(list(map(lambda i: i**2, list(range(1, n + 1)))))
