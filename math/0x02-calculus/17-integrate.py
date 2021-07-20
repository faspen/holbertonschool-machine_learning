#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """Integrate function"""

    new = [C]

    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if poly == [0]:
        return new
    for i in range(len(poly)):
        if isinstance(poly[i], int) or isinstance(poly[i], float):
            new.append(poly[i] / (i + 1))
        else:
            return None
