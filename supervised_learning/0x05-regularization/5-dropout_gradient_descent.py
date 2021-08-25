#!/usr/bin/env python3
"""Dropout gradient descent"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update weights with dropout and gradient descent"""
    m = Y.shape[1]
    for i in reversed(range(L)):
        A = cache['A' + str(i + 1)]
        A2 = cache['A' + str(i)]
        if i == L - 1:
            dz = A - Y
            W = weights['W' + str(i + 1)]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da * cache['D{}'.format(i + 1)]
            dz /= keep_prob
            W = weights['W' + str(i + 1)]
        dw = np.matmul(A2, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i + 1)] = weights['W' + str(i + 1)] - alpha * dw.T
        weights['b' + str(i + 1)] = weights['b' + str(i + 1)] - alpha * db
