#!/usr/bin/env python3
"""Positional encoding task"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positionaal encoding for a transformer"""
    pos_embeddings = np.zeros((max_seq_len, dm))

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            power = np.exp(j * -np.log(10000.0) / dm)
            pos_embeddings[i, j] = (np.sin(i * power))
            pos_embeddings[i, j + 1] = (np.cos(i * power))

    return pos_embeddings
