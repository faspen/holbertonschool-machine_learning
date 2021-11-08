#!/usr/bin/env python3
"""Baum welch algorithm"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):

