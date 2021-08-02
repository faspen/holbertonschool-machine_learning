#!/usr/bin/env python3
"""Neuron class"""


import numpy as np


class Neuron():
    """class for neuron"""

    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter W"""
        return self.__W

    @ property
    def b(self):
        """Getter b"""
        return self.__b

    @ property
    def A(self):
        """Getter A"""
        return self.__A
