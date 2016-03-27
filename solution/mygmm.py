#!/usr/bin/env python
# encoding: utf-8

"""
mygmm.py
 
Created by Shuailong on 2016-03-26.

Gaussian Mixture Model: Representation of a Gaussian mixture model probability distribution.

Reference: 
    1. http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM

Issues:
    1. math domain error when calculate log. 
    

"""

import numpy as np
from random import random
from math import log, exp, pow, pi, sqrt

class GMM:
    def __init__(self, n_components=1, covariance_type='spherical', tol=1e-3, n_iter=100, n_init=1):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.n_iter = n_iter
        self.n_init = n_init

        if self.covariance_type not in ['spherical']:
            raise ValueError('The specified covariance_type is not supported yet!')

    def _gaussian(self, x, miu, sigma_sq):
        C = 1.0/sqrt(pow(2*pi, len(x))*sigma_sq)
        return C*exp(-0.5/sigma_sq*np.inner(x-miu, x-miu))


    def fit(self, X):
        d = len(X[0])
        n = len(X)

        print'%s datapoints loaded.\n'%n

        self.converged_ = False

        mins = [min(X[:,i]) for i in range(d)]
        maxs = [max(X[:,i]) for i in range(d)]

        bestL = -float('inf')
        for iteration in range(self.n_init):
            print 'The %sth EM'%(iteration+1)

            # initialize weights
            weights_ = [1.0/self.n_components]*self.n_components

            # initialize variances
            mean = [np.mean(X[:, i]) for i in range(d)]
            sigma_sq = 1.0/(d*n)*sum([np.inner(x - mean, x - mean) for x in X])
            covars_ = [sigma_sq] * self.n_components
            # print self.covars_

            # initialize means
            means_ = [[random()*(maxs[j]-mins[j]) + mins[j] for j in range(d)] for k in range(self.n_components)]
            # print self.means_

            # p: n_components * n
            p = [[0 for i in range(n)] for j in range(self.n_components)]
            lastL = -float('inf')
            L = -float('inf')
            for j in range(self.n_iter):
                print 'Iteration', j+1, ':'
                # E-step
                s = sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components) for t in range(n)])
                
                for t in range(n):
                    for i in range(self.n_components):
                        p[i][t] = weights_[i]*self._gaussian(X[t], means_[i], covars_[i])/s
                
                # M-step
                n_cap = [sum([p[i][t] for t in range(n)])*n for i in range(self.n_components)]
                weights_ = [n_cap[i]/float(n) for i in range(self.n_components)]
                means_ = [sum([np.multiply(1/float(weights_[i])*p[i][t], X[t]) for t in range(n)]) for i in range(self.n_components)]
                covars_ = [1.0/d/n_cap[i]*sum([p[i][t]*np.inner(X[t]-means_[i], X[t]-means_[i]) for t in range(n)]) for i in range(self.n_components)]

                L = sum([log(sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components)])) for t in range(n)])
                print 'Likelyhood:', L

                if L - lastL >= 0 and L - lastL < self.tol:
                    break
                lastL = L

                print

            if L > bestL:
                bestL = L
                self.weights_ = weights_
                self.means_ = means_
                self.covars_ = covars_

            print
        print 'Max L:', bestL
        print 'Weights:', self.weights_
        print 'Means:', self.means_
        print 'Variances:', self.covars_

    def score(self, X):
        pass

def main():
    X = np.loadtxt(open('../data/data.txt',"rb"),delimiter=" ",skiprows=0)
    clf = GMM(n_components=2, n_init=1)
    clf.fit(X)


if __name__ == '__main__':
    main()

