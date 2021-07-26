#!/usr/bin/env python3
"""Expo"""


class Exponential():
    """class for expo"""

    def __init__(self, data=None, lambtha=1.):
        """Init for expo"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / float(sum(data)) / len(data)
