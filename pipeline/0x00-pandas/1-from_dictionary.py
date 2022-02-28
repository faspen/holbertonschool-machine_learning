#!/usr/bin/env python3
"""From dictionary task"""


import pandas as pd


def from_dictionary():
    """Create pandas dataframe from dictionary"""
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD")
    )

    return df
