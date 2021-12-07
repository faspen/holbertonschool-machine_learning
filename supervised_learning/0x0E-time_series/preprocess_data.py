#!/usr/bin/env python3
"""Preprocess Data"""


import pandas as pd


def preprocess_data(old_data, clean_data):
    """Preprocess time series data"""
    cd = pd.read_csv(old_data)
    cleansed = cd.dropna(axis=0)
    default = (cleansed - cleansed.mean()) / cleansed.std()
    default["Timestamp"] = cleansed["Timestamp"]
    proc = default[["Timestamp", "Volume_(Currency)", "Weighted_Price"]]
    proc.to_csv(clean_data)
    proc.head()
