#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""


import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Perform BO on noiseless 1D Gaussian Process"""

