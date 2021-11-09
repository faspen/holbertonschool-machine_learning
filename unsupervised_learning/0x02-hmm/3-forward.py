#!/usr/bin/env python3
"""Forward algorithm"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Perform forward algorithm"""
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

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for i in range(1, T):
        for j in range(N):
            tmp = F[:, i - 1] * Transition[:, j]
            F[j, i] = np.sum(tmp * Emission[j, Observation[i]])

    P = np.sum(F[:, -1])

    return P, F
