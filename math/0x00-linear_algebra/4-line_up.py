#!/usr/bin/env python3
"""Add two arrays"""


def add_arrays(arr1, arr2):
    """Add two arrays method"""
    answer = []
    if len(arr1) != len(arr2):
        return None
    else:
        for n1, n2 in zip(arr1, arr2):
            answer.append(n1 + n2)
    return answer
