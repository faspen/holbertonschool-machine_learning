#!/usr/bin/env python3
"""Hello, sklearn"""


import sklearn.cluster


def kmeans(X, k):
    """Perform K-means on dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
