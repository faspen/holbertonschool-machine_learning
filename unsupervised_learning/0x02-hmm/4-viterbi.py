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

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    """alpha = np.zeros((N, T))
    tmp = (Initial * Emission[:, Observation[0]].reshape(-1, 1))
    alpha[:, 0] = tmp.reshape(-1)

    backtrack = np.zeros((N, T))"""

    return path, P
