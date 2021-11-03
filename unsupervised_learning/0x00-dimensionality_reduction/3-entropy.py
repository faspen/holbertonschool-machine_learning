#!/usr/bin/env python3
"""Entropy module"""


import numpy as np


def HP(Di, beta):
    """Calculate Shannon entropy"""
    power = np.exp(-Di * beta)
    total = np.sum(power)
    Pi = power / total
    Hi = -np.sum(Pi * np.log2(Pi))

    return (Hi, Pi)
