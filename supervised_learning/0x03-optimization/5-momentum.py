#!/usr/bin/env python3
"""Momentum task"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Return updated variable and new moment"""
    moment = beta1 * v + (1 - beta1) * grad
    update = var - alpha * moment
    return update, moment
