#!/usr/bin/env python3
"""GMM task module"""


import sklearn.mixture


def gmm(X, k):
    """Calculate GMM from a dataset"""
    GM = sklearn.mixture.GaussianMixture(n_components=k)
    par = GM.fit(X)
    pi = par.weights_
    m = par.means_
    S = par.covariances_
    clss = GM.predict(X)
    bic = GM.bic(X)

    return pi, m, S, clss, bic
