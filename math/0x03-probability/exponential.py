#!/usr/bin/env python3
"""Expo module"""


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
            self.lambtha = float(1 / ((sum(data)) / len(data)))

    def pdf(self, x):
        """PDF function"""
        if x < 0:
            return 0
        else:
            return (self.lambtha * 2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        """CDF function"""
        if x < 0:
            return 0
        else:
            return (1 - 2.7182818285 ** (-self.lambtha * x))
