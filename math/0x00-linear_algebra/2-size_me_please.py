#!/usr/bin/env python3
"""Size matrix"""


def matrix_shape(matrix):
    """Get size of matrix"""
    result = []
    if not matrix:
        result = [0]
        return result
    else:
        result.append(len(matrix))
        while isinstance(matrix[0], list):
            result.append(len(matrix[0]))
            matrix = matrix[0]
        return result
