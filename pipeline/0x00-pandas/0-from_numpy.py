#!/usr/bin/env python3
"""From numpy task"""

import pandas as pd


def from_numpy(array):
    """Makes dataframe from array"""
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H",
                "I", "J", "K", "L", "M", "N", "O", "P",
                "Q", "R", "S", "T", "U", "V", "W", "X",
                "Y", "Z"]
    labels = []

    for i in range(len(array[0])):
        labels.append(alphabet[i])
    df = pd.DataFrame(array, columns=labels)

    return df
