#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine if gradient descent should stop"""
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        boolean = True
    else:
        boolean = False
    return boolean, count
