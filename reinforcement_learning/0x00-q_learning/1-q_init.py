#!/usr/bin/env python3
"""Initialize the q_table"""


import gym
import numpy as np


def q_init(env):
    """Initialize qtable"""
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    return q_table
