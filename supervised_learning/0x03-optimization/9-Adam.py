#!/usr/bin/env python3
"""Adam task"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Return updated var, new moment, second new moment"""
    mw = beta1 * v + (1 - beta1) * grad
    vw = beta2 * s + (1 - beta2) * (grad ** 2)

    mw2 = mw / (1 - beta1 ** t)
    vw2 = vw / (1 - beta2 ** t)

    var = var - alpha * (mw2 / (np.sqrt(vw2) + epsilon))

    return var, mw, vw
