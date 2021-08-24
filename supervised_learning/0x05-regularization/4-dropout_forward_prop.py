#!/usr/bin/env python3
"""Forward prop with dropout"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return dictionary of outputs from each layer"""
    cache = {}
    cache['A0'] = X

    for i in range(L):
        di = np.matmul(weights['W' + str(i + 1)],
                       cache['A' + str(i)]) + weights['b' + str(i + 1)]

        if i == L - 1:
            cache['A' + str(i + 1)] = np.exp(di) / \
                (np.sum(np.exp(di), axis=0, keepdims=True))
        else:
            cache['A' + str(i + 1)] = np.tanh(di)
            ran = np.random.rand(
                cache['A' + str(i + 1)].shape[0],
                cache['A' + str(i + 1)].shape[1]) < keep_prob
            delete = np.where(ran == 1, 1, 0)
            cache['A' + str(i + 1)] *= delete
            cache['A' + str(i + 1)] /= keep_prob
            cache['D' + str(i + 1)] = delete
    return cache
