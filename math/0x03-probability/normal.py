#!/usr/bin/env python3
"""Normal"""


class Normal():
    """Normal class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """init for normal"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            step = [(x - self.mean) ** 2 for x in data]
            self.stddev = pow((sum(step) / len(data)), 0.5)

    def z_score(self, x):
        """Get z score"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Get x value"""
        return ((z * self.stddev) + self.mean)

    def pdf(self, x):
        """Get PDF"""
        i = (1 / (self.stddev * ((2 * 3.14159265)**(1 / 2))))
        j = (-1 / 2) * ((x - self.mean) / self.stddev)**2
        return i * (2.71828**j)

    def cdf(self, x):
        """Get CDF"""
        i = (x - self.mean) / (self.stddev * (2 ** (1 / 2)))
        erf_1 = (2 / 3.14159265 ** (1 / 2))
        erf_2 = (i - ((i**3) / 3) + ((i**5) / 10) -
                 ((i**7) / 42) + ((i**9) / 216))
        val = (1 / 2) * (1 + erf_1 * erf_2)
        return val
