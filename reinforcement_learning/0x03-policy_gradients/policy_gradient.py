#!/usr/bin/env python3


import numpy as np
import gym


def policy(matrix, weight):
    """Computes policy with the weight of a matrix"""
    mult = matrix.dot(weight)
    exp = np.exp(mult)
    policy = exp / np.sum(exp)

    return policy


def policy_gradient(state, weight):
    """Compute monte carlo policy gradient"""
    PG = policy(state, weight)

    action = np.random.choice(len(PG[0]), p=PG[0])
    rs = PG.reshape(-1, 1)

    softmax = (np.diagflat(rs) - np.dot(rs, rs.T))[action, :]
    tmp = softmax / PG[0, action]
    gradient = state.T.dot(tmp[None, :])

    return action, gradient
