#!/usr/bin/env python3
"""Cats got your tongue"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Cat two strings with numpy"""
    return np.concatenate((mat1, mat2), axis)
