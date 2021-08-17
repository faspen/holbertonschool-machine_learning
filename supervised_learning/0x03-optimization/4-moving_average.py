#!/usr/bin/env python3
"""Moving average"""


import numpy as np


def moving_average(data, beta):
    """Return moving averages of data"""
    weight = []
    n = 0
    for i in range(len(data)):
        n = beta * n + (1 - beta) * data[i]
        weight.append(n / (1 - beta ** (i + 1)))

    return weight
