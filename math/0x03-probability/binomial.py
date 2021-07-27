#!/usr/bin/env python3
"""Binomial"""


class Binomial():
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Init"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            total = sum([((i - mean) ** 2) for i in data])
            res = (total / len(data))
            exp = 1 - (res / mean)
            if ((mean / exp) - (mean // exp)) >= 0.5:
                self.n = 1 + int(mean + exp)
            else:
                self.n = int(mean / exp)
            self.p = float(mean / self.n)

    def factorial(self, k):
        """Shortcut"""
        if k in [0, 1]:
            return 1
        return k * self.factorial(k - 1)

    def pmf(self, k):
        """Get pmf"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k is None:
            return 0
        return (self.factorial(self.n) / (self.factorial(k) *
                self.factorial(self.n - k))
                ) * ((self.p ** k) * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Get cdf"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k is None:
            return 0
        return sum([(self.factorial(self.n) / (self.factorial(x) *
                   self.factorial(self.n - x)))
            * ((self.p ** x) * (1 - self.p) ** (self.n - x))
            for x in range(k + 1)])
