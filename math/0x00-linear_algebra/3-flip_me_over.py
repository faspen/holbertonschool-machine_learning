#!/usr/bin/env python3
"""Transpose module"""


def matrix_transpose(matrix):
    """Transpose method"""
    transpose = []
    transpose = [[matrix[j][i]
                  for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transpose
