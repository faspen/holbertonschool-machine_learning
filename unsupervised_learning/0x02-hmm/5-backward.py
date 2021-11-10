#!/usr/bin/env python3
"""Backward algorithm"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Perform backward algorithm"""
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    sums = np.sum(Emission, axis=1)
    if not (sums == 1.0).all():
        return None, None
    sums = np.sum(Transition, axis=1)
    if not (sums == 1.0).all():
        return None, None
    sums = np.sum(Initial, axis=0)
    if not (sums == 1.0).all():
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    for i in range(T - 2, -1, -1):
        for j in range(N):
            B[j, i] = np.sum(B[:, i + 1] * Transition[j, :] *
                             Emission[:, Observation[i + 1]])
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
