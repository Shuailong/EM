#!/usr/bin/env python
# encoding: utf-8

"""
main.py
 
Created by Shuailong on 2016-03-26.

Main entry of Machine Learining Assignment 3.

"""

import matplotlib.pyplot as plt
import numpy as np

from mygmm import GMM

# from sklearn.mixture import GMM

def main():
    X = np.loadtxt(open('../data/data.txt',"rb"),delimiter=" ",skiprows=0)
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    clf = GMM(n_components=2, n_init=1, tol=1e-6, n_iter=100)
    clf.fit(X)


if __name__ == '__main__':
    main()

    