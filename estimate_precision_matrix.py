__author__ = "Lingqing Gan"

#------------------------------------------------------------------------------------

#   FILE_NAME:      estimate_precision_matrix.py

#   DESCRIPTION:    Calculate the precision matrix given the multi-variate time-series
#                   signals.

#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           06/15/2018

#------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix


# set up parameters

T = 300                   # number of time slots
N = 3                       # number of agents


# mean of nodes' random signals

mean = np.zeros(N)


# draw a random covariance matrix (symmetric positive definite)

C = make_spd_matrix(N)
print(C)


# generate random signal for time length T

y = [np.random.multivariate_normal(mean, C) for t in range(0,T)]
y = np.matrix(y)

# calculate the mean of y

y_mean = np.mean(y)


# calculate covariance matrix

temp = np.zeros((N,N))

for i in range(0,T):

    temp2 = y[i]-y_mean
    incre = temp2.T * temp2
    temp = temp + incre


Q = 1/(T-1) * temp

print(Q)


















