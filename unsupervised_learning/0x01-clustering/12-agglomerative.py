#!/usr/bin/env python3
"""Agglomerative module"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering"""
    h = scipy.cluster.hierarchy
    link = h.linkage(X, method='ward')
    clss = h.fcluster(link, t=dist, criterion='distance')

    plt.figure()
    h.dendrogram(link, color_threshold=dist)
    plt.show()

    return clss
