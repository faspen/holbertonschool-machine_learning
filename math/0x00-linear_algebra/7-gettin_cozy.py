#!/usr/bin/env python3
"""Gettin Cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concat two matrices along specific axis"""
    new = []

    if (len(mat1[0]) == len(mat2[0])) and (axis == 0):
        new = [r.copy() for r in mat1]
        new += [r.copy() for r in mat2]
        return new
    elif (len(mat1) == len(mat2)) and (axis == 1):
        new = [mat1[r] + mat2[r] for r in range(len(mat1))]
        return new
    else:
        return None
