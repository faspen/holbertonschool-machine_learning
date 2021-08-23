#!/usr/bin/env python3
"""Create confusion"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """Return numpy array (classes, classes)"""
    return np.matmul(labels.T, logits)
