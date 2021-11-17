#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Perform BO on noiseless 1D Gaussian Process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Class init constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate next best sample location"""
        pr1, pr2 = self.gp.predict(self.X_s)

        if self.minimize is False:
            tmp = np.amax(self.gp.Y)
            res = pr1 - tmp - self.xsi
        else:
            tmp = np.amin(self.gp.Y)
            res = tmp - pr1 - self.xsi

        zer = np.zeros(pr2.shape)
        for i in range(len(pr2)):
            if pr2[i] != 0:
                zer[i] = res[i] / pr2[i]
            else:
                zer[i] = 0
        EI = res * norm.cdf(zer) + pr2 * norm.pdf(zer)
        EI[pr2 == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimize the black box"""
        pos = []

        for i in range(iterations):
            fx, ei = self.acquisition()
            fy = self.f(fx)
            amx = np.argmax(ei)
            if amx in pos:
                pos.append(np.argmax(ei))
                break
            self.gp.update(fx, fy)
            pos.append(np.argmax(ei))

        if self.minimize is True:
            next = np.argmin(self.gp.Y)
        else:
            next = np.argmax(self.gp.Y)

        X_opt = self.gp.X[next]
        Y_opt = self.gp.Y[next]

        return X_opt, Y_opt
