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
            num = poly[i] / (i + 1)
            if float.is_integer(num):
                num = int(num)
            new.append(num)
        else:
            return None
    return new
