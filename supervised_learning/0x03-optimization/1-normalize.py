#!/usr/bin/env python3
"""Normalize matrix"""


def normalize(X, m, s):
    """Normalize the matrix"""
    norm = (X - m) / s
    return norm
