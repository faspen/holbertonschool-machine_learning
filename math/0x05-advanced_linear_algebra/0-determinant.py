#!/usr/bin/env python3
"""Determinant calculation"""


def determinant_helper(matrix, total=0):
    """Assist in the calculation of determinant"""
    indices = list(range(len(matrix)))

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1])

    for fc in indices:
        tmp = matrix.copy()
        tmp = matrix[1:]
        rlen = len(tmp)

        for r in range(rlen):
            tmp[r] = tmp[r][0:fc] + tmp[r][fc + 1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant_helper(tmp)
        total += sign * matrix[0][fc] * sub_det

    return total


def determinant(matrix):
    """Return determinant of matrix"""
    if matrix == [[]]:
        return 1

    if isinstance(matrix, list) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    indices = list(range(len(matrix)))

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    return determinant_helper(matrix)
