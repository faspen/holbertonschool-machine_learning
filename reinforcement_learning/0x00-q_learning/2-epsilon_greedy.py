#!/usr/bin/env python3
"""Epsilon Greedy"""


import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """Use epsilon greedy to determine next action"""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])

    return action
