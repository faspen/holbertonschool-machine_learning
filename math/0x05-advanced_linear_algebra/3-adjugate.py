#!/usr/bin/env python3
"""Adjugate calculator"""


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


def cofactor(matrix):
    """Return cofactor matrix of matrix"""
    m = minor(matrix)
    cof = m.copy()

    for i in range(len(m)):
        for j in range(len(m)):
            cof[i][j] *= (-1)**(i + j)
    return cof


def transpose(matrix):
    """Transpose given matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def adjugate(matrix):
    """Return adjugate matrix of matrix"""
    cof = cofactor(matrix)
    return transpose(cof)
