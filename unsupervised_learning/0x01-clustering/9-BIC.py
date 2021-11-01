#!/usr/bin/env python3
"""BIC task module"""


import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for GMM"""

