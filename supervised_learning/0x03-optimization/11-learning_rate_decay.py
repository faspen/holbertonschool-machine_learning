#!/usr/bin/env python3
"""Learning rate decay"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return updated value for alpha"""
    step = global_step // decay_step
    updated = alpha / (1 + decay_rate * step)
    return updated
