#!/usr/bin/env python3
"""Specificity module"""


import numpy as np


def specificity(confusion):
    """Return array shape (classes,)"""
    fp = confusion.sum(axis=0) - np.diag(confusion)
    fn = confusion.sum(axis=1) - np.diag(confusion)
    tp = np.diag(confusion)
    tn = confusion.sum() - (fp + fn + tp)
    res = tn / (tn + fp)
    return res
