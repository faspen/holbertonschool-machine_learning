#!/usr/bin/env python3
"""EM task module"""


import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform expectaion maximization for GMM"""

