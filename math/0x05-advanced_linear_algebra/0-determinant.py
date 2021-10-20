#!/usr/bin/env python3
"""Determinant calculation"""


"""def determinant_helper(matrix, total=0):
    """"""Assist in the calculation of determinant""""""
    indices = list(range(len(matrix)))

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1])

    row = matrix[0]
    cof = 1
    for i in range(len(matrix[0])):
        tmp = [m[:] for m in matrix]
        del tmp[0]
        for val in tmp:
            del val[i]
        total += row[i] * determinant_helper(tmp) * cof
        cof *= -1

    return total"""


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

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1])

    row = matrix[0]
    total = 0
    cof = 1
    for i in range(len(matrix[0])):
        tmp = [m[:] for m in matrix]
        del tmp[0]
        for val in tmp:
            del val[i]
        total += row[i] * determinant(tmp) * cof
        cof *= -1

    return total
