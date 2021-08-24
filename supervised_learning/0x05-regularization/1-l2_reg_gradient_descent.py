#!/usr/bin/env python3
"""Gradient descent"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update w and b using gradient descent with l2"""
    w_proxy = weights.copy()
    for i in range(L, 0, -1):
        m = Y.shape[1]
        if i != L:
            di = np.multiply(np.matmul(
                w_proxy['W' + str(i + 1)].T, di), 1 - cache['A' + str(i)] ** 2)
        else:
            di = cache['A' + str(i)] - Y
        dwi = np.matmul(di, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(di, axis=1, keepdims=True) / m
        l2 = (1 - alpha * lambtha / m)
        weights['W' + str(i)] = l2 * w_proxy['W' + str(i)] - alpha * dwi
        weights['b' + str(i)] = w_proxy['b' + str(i)] - alpha * dbi
