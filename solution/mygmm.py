#!/usr/bin/env python
# encoding: utf-8

"""
mygmm.py
 
Created by Shuailong on 2016-03-26.

Gaussian Mixture Model: Representation of a Gaussian mixture model probability distribution.

Reference: 
    1. http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM

Issues:
    1. _gaussian C: division by zero error may occur. 

"""

import numpy as np
from random import random
from math import log, exp, pow, pi, sqrt
import matplotlib.pyplot as plt
import matplotlib
from warnings import warn

class GMM:
    def __init__(self, n_components=1, covariance_type='spherical', tol=1e-3, n_iter=100, n_init=1):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.n_iter = n_iter
        self.n_init = n_init

        if self.covariance_type not in ['spherical']:
            raise ValueError('The specified covariance_type is not supported yet!')

        if self.n_components > 8:
            warn('Too many clusters, not enough colors to visualize!')

    def _gaussian(self, x, miu, sigma_sq):
        C = 1.0/pow(2*pi*sigma_sq, len(x)/2.0)
        return C*exp(-0.5/sigma_sq*np.inner(x-miu, x-miu))


    def fit(self, X):
        d = len(X[0])
        n = len(X)

        print'%s datapoints loaded.\n'%n

        self.converged_ = False

        mins = [min(X[:,i]) for i in range(d)]
        maxs = [max(X[:,i]) for i in range(d)]

        bestL = -float('inf')
        bestP = np.zeros(shape=(self.n_components, n))

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
            p = np.zeros(shape=(self.n_components, n))
            lastL = -float('inf')
            L = -float('inf')
            for j in range(self.n_iter):
                print 'Iteration', j+1, ':'
                # E-step
                for t in range(n):
                    for i in range(self.n_components):
                        p[i][t] = weights_[i]*self._gaussian(X[t], means_[i], covars_[i])\
                            /sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components)])

                # M-step
                n_cap = [sum([p[i][t] for t in range(n)]) for i in range(self.n_components)]
                weights_ = [n_cap[i]/float(n) for i in range(self.n_components)]
                means_ = [sum([p[i][t] * X[t] for t in range(n)])/n_cap[i] for i in range(self.n_components)]
                # print means_
                covars_ = [1.0/d/n_cap[i]*sum([p[i][t]*np.inner(X[t]-means_[i], X[t]-means_[i]) for t in range(n)]) for i in range(self.n_components)]
                # print covars_
                L = sum([log(sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components)])) for t in range(n)])
                print 'Likelyhood:', L

                if L - lastL >= 0 and L - lastL < self.tol:
                    self.converged_ = True
                    break
                lastL = L

                print

            if L > bestL:
                bestL = L
                bestP = p
                self.weights_ = weights_
                self.means_ = means_
                self.covars_ = covars_
            print

        print 'Converged:', self.converged_
        print 'Max L:', bestL
        print 'Weights:', self.weights_
        print 'Means:', self.means_
        print 'Variances:', self.covars_

        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white']

        plt.gca().set_aspect('equal', adjustable='box')
        # draw data points
        labels = [np.argmax(bestP[:, t]) for t in range(n)]
        # plt.scatter(X[:,0], X[:,1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

        for i in range(self.n_components):
            X_i = np.asarray([X[j] for j in range(n) if labels[j] == i])
            plt.scatter(X_i[:,0], X_i[:,1], c=colors[i])

        # draw circles
        for i in range(self.n_components):
            circle = plt.Circle(self.means_[i], sqrt(self.covars_[i]), color=colors[i], fill=False)
            plt.gcf().gca().add_artist(circle)

        # if self.converged_:
        plt.show()


    def score(self, X):
        pass

def main():
    X = np.loadtxt(open('../data/data.txt',"rb"),delimiter=" ",skiprows=0)
    clf = GMM(n_components=2, n_init=1, tol=1e-6, n_iter=100)
    clf.fit(X)

if __name__ == '__main__':
    main()

