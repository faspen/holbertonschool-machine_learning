#!/usr/bin/env python3


import pandas as pd


def from_file(filename, delimiter):
    """Make dataframe from a file"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
