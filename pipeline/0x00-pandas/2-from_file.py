#!/usr/bin/env python3


import pandas as pd


def from_file(filename, delimiter):
    """Make dataframe from a file"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
print(df2.tail())