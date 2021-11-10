#!/usr/bin/env python3
"""Baum welch algorithm"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Perform Baum Welch algorithm"""
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

    return Transition, Emission
