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
import time

from images2gif import writeGif
from PIL import Image
import os

class GMM:
    def __init__(self, n_components=1, covariance_type='spherical', tol=1e-3, n_iter=100, n_init=1, verbose=False, soft=True, image=True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.n_iter = n_iter
        self.n_init = n_init
        self.verbose = verbose
        self.soft = soft
        self.image = image

        if self.covariance_type not in ['spherical']:
            raise ValueError('The specified covariance_type is not supported yet!')

        if self.n_components > 8:
            warn('Too many clusters, not enough colors to visualize!')

        self._colors = ['red', 'blue', 'black', 'cyan', 'magenta', 'yellow', 'green', 'white'][:self.n_components]

    def _gaussian(self, x, miu, sigma_sq):
        C = 1.0/pow(2*pi*sigma_sq, len(x)/2.0)
        return C*exp(-0.5/sigma_sq*np.inner(x-miu, x-miu))


    def fit(self, X):
        if self.image:
            # remove old image files
            filelist = ["../output_image/"+f for f in os.listdir("../output_image/")]
            for f in filelist:
                os.remove(f)

        d = len(X[0])
        n = len(X)

        print'%s datapoints loaded.'%n

        self.converged_ = False

        mins = [min(X[:,i]) for i in range(d)]
        maxs = [max(X[:,i]) for i in range(d)]

        bestL = -float('inf')
        bestP = np.zeros(shape=(self.n_components, n))

        for iteration in range(self.n_init):
            if self.verbose:
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
            Ls = []
            for iter_ in range(self.n_iter):
                if self.verbose:
                    print 'Iteration', iter_+1, ':'
                # E-step
                for t in range(n):
                    if self.soft:
                        for i in range(self.n_components):
                            p[i][t] = weights_[i]*self._gaussian(X[t], means_[i], covars_[i])\
                                /sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components)])
                    else:
                        label = np.argmax([weights_[i]*self._gaussian(X[t], means_[i], covars_[i]) for i in range(self.n_components)])
                        p[label][t] = 1

                # M-step
                n_cap = [sum([p[i][t] for t in range(n)]) for i in range(self.n_components)]
                weights_ = [n_cap[i]/float(n) for i in range(self.n_components)]
                means_ = [sum([p[i][t] * X[t] for t in range(n)])/n_cap[i] for i in range(self.n_components)]
                # print means_
                covars_ = [1.0/d/n_cap[i]*sum([p[i][t]*np.inner(X[t]-means_[i], X[t]-means_[i]) for t in range(n)]) for i in range(self.n_components)]
                # print covars_
                L = sum([log(sum([weights_[j]*self._gaussian(X[t], means_[j], covars_[j]) for j in range(self.n_components)])) for t in range(n)])
                Ls.append(L)
                if self.verbose:
                    print 'Likelyhood:%s\n'%L

                if self.image:
                # draw data points
                    labels = [np.argmax(p[:, t]) for t in range(n)]
                    plt.figure(1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.scatter(X[:,0], X[:,1], s=12, facecolors='none', c=labels, cmap=matplotlib.colors.ListedColormap(self._colors))
                    # draw circles
                    for i in range(self.n_components):
                        circle = plt.Circle(means_[i], sqrt(covars_[i]), color=self._colors[i], fill=False)
                        plt.gcf().gca().add_artist(circle)
                        plt.text(means_[i][0], means_[i][1], str(round(weights_[i], 3)), fontsize=7, color=self._colors[i])
                    figname = '../output_image/iteration'+ str(iter_+1) +'.png'
                    plt.title('Iteration ' + str(iter_+1))
                    plt.savefig(figname, dpi=300)
                    plt.close()

                if L - lastL >= 0 and L - lastL < self.tol:
                    self.converged_ = True
                    break
                lastL = L

            if self.image:
                plt.figure(2)
                plt.plot(range(len(Ls)), Ls)
                plt.title('Likelihood')
                plt.savefig('../output_image/L.png', dpi=300)
                plt.close()

            if L > bestL:
                bestL = L
                bestP = p
                self.weights_ = weights_
                self.means_ = means_
                self.covars_ = covars_

        print '---------------------------'
        print 'Converged:', self.converged_
        print 'Max L:', bestL
        print 'Weights:', self.weights_
        print 'Means:', self.means_
        print 'Variances:', self.covars_

        if self.image:
            # draw data points
            labels = [np.argmax(bestP[:, t]) for t in range(n)]
            plt.figure(3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.scatter(X[:,0], X[:,1], s=12, facecolors='none', c=labels, cmap=matplotlib.colors.ListedColormap(self._colors))

            # draw circles
            for i in range(self.n_components):
                circle = plt.Circle(self.means_[i], sqrt(self.covars_[i]), color=self._colors[i], fill=False)
                plt.gcf().gca().add_artist(circle)
                plt.text(self.means_[i][0], self.means_[i][1], str(round(self.weights_[i], 3)), fontsize=7, color=self._colors[i])
            plt.title('Final result')
            plt.savefig('../output_image/best.png', dpi=300)
            plt.close()

            # make a gif image
            file_names = ['../output_image/'+fn for fn in os.listdir('../output_image/') if fn.startswith('iteration')]
            file_names.sort(key=lambda x: int(x[len('../output_image/iteration'):-len('.png')]))
            images = [Image.open(fn) for fn in file_names]
            writeGif("../output_image/movie.gif", images, duration=0.25)

    def score(self, X):
        pass

def main():
    X = np.loadtxt(open('../data/data.txt',"rb"),delimiter=" ",skiprows=0)
    clf = GMM(n_components=2, n_init=1, tol=1e-6, n_iter=200, verbose=True, image=True)
    clf.fit(X)

if __name__ == '__main__':
    main()

