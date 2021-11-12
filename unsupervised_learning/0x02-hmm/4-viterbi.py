#!/usr/bin/env python3
"""Viretbi algorithm"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculate likely sequence of states"""
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

    N = Emission.shape[0]
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    alpha = np.zeros((N, T))
    beta = np.zeros((N, T))
    path = []
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for i in range(1, T):
        for j in range(N):
            beta[j, i] = np.argmax(
                Emission[j, Observation[i]] * Transition[:, j] *
                alpha[:, i - 1])
            alpha[j, i] = np.max(Emission[j, Observation[i]]
                                 * Transition[:, j] * alpha[:, i - 1])

    P = np.max(alpha[:, -1], axis=0)
    path.append(np.argmax(alpha[:, -1], axis=0))

    for i in range(T - 1, 0, -1):
        path.insert(0, int(beta[int(path[0]), i]))

    return path, P
