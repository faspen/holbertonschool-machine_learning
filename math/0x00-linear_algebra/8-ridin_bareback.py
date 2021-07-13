#!/usr/bin/env python3
"""Ridin bareback"""


def mat_mul(mat1, mat2):
    """Mult two matrices"""
    r1 = len(mat1)
    c1 = len(mat1[0])
    r2 = len(mat2)
    c2 = len(mat2[0])

    if (c1 != r2):
        return None
    else:
        result = []
        for i in range(r1):
            tmp_row = []
            for j in range(c2):
                r_sum = []
                for k in range(c1):
                    r_sum.append(mat1[i][k] * mat2[k][j])
                tmp_row.append(sum(r_sum))
            result.append(tmp_row)
        return result
