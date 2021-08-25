#!/usr/bin/env python3
"""Slice like a ninja"""


def np_slice(matrix, axes={}):
    """Return new array"""
    pieces = []
    for i in range(max(axes) + 1):
        if i in axes.keys():
            pieces.append(slice(*axes.get(i)))
        else:
            pieces.append(slice(None, None, None))
    return matrix[tuple(pieces)]
