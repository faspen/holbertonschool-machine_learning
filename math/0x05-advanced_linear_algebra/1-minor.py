#!/usr/bin/env python3
"""Minor calculator"""


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


def minor_helper(matrix, i, j):
    """Assist in the calculation of matrix minor"""
    answer = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
    return determinant(answer)


def minor(matrix):
    """Return minor of matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if isinstance(matrix, list) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minors_list = []
    for row in range(len(matrix)):
        minors = []
        for col in range(len(matrix)):
            minors.append(minor_helper(matrix, row, col))
        minors_list.append(minors)

    return minors_list
