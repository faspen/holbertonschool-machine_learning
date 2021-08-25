#!/usr/bin/env python3
"""The Whole Barn"""


def matrix_shape(matrix):
    """Check the shape"""
    shape = [len(matrix)]
    while isinstance(matrix[0], list):
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def add_matrices(mat1, mat2):
    """Add two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(matrix_shape(mat1)) == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    elif len(matrix_shape(mat1)) == 2:
        return [[mat1[i][j] + mat2[i][j]
                 for j in range(len(mat1[0]))] for i in range(len(mat1))]
    elif len(matrix_shape(mat1)) >= 3:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
