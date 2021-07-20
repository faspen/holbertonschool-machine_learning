#!/usr/bin/env python3
"""Matisse"""


def poly_derivative(poly):
    """Derivative"""
    der = []
    if not isinstance(poly, list):
        return None
    elif len(poly) is 0:
        return None
    else:
        for i in range(len(poly)):
            if isinstance(poly[i], int):
                if i is not 0:
                    der.append(poly[i] * i)
            else:
                return None
        if sum(der) is 0:
            return [0]
        else:
            return der
