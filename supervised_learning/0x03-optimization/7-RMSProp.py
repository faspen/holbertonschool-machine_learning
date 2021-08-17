#!/usr/bin/env python3
"""RMS prop"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Return update and moment"""
    moment = beta2 * s + (1 - beta2) * (grad ** 2)
    update = var - alpha * grad / (moment ** (1 / 2) + epsilon)
    return update, moment
