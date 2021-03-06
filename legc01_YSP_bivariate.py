__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      YSP_bi-variate.py

#   DESCRIPTION:    Gibbs sampling for bi-variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 6.

#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           07/01/2018 -

# ------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import wishart, multivariate_normal, bernoulli, invwishart, beta
from scipy.stats import gamma as sci_gamma
from math import *
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------
#
#  function definition and invocation
#
# ---------------------------------------------------------------------------------------


def book_keeping_n(z):

    n = np.array([])
    for i in range(0, len(z)):
        if i == 0:
            n_count = 1
            temp = z[i]
            continue
        elif i < len(z) - 1:
            if z[i] == temp:
                n_count = n_count + 1
            else:
                n = np.append(n, n_count)
                n_count = 1
            temp = z[i]
            continue
        else:
            if z[i] == temp:
                n_count = n_count + 1
                n = np.append(n, n_count)
            else:
                n = np.append(n, n_count)
                n = np.append(n, 1)
            continue

    return n


def book_keeping_z(z):

    flag1 = 0
    flag2 = 0

    for i in range(0, len(z)):
        if z[i] == -10:
            flag1 = 1
            new = i

    for i in range(1, len(z)):
        if z[i] - z[i-1] == 2:
            flag2 = 1
            gap = i

    if flag1 == 1 and flag2 == 0:
        z[new] = z[new-1] + 1

        if new < len(z) - 1 and z[new] == z[new+1]:

            for i in range(new+1, len(z)):
                z[i] = z[i] + 1

    if flag2 == 1 and flag1 == 0:
        for i in range(gap, len(z)):
            z[i] = z[i]-1

    for i in range(0, len(z)-1):
        if z[i+1] > z[i] + 1:
            print("error")
            break

    return z


def multi_student_pdf(x, loc, scale, df):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    d = len(x)
    num = gamma(1. * (d+df)/2)
    denom1 = gamma(1.*df/2) * pow(df*pi, 1.*d/2) * pow(np.linalg.det(scale), 1./2)
    denom2 = (1 + (1./df)*np.dot(np.dot((x - loc),np.linalg.inv(scale)), (x - loc)))**int((d+df)/2)
    denom = denom1 * denom2

    denom = float(denom)

    d = 1. * num / denom
    return d


# hyper-parameter

a = 1
b = 1

alpha = np.random.gamma(a, scale=1/b)                # shape, scale, b is rate

T = 1000

V0 = np.matrix([[1, 0], [0, 1]])

n0 = 3

v = 2                                               # degree of freedom

# ---------------------------------------------------------------------------------------
#
# Generative Model:
#
# generate random signals according to Yule-Simon process
#
# ---------------------------------------------------------------------------------------

s = np.zeros(T)

x = np.zeros(T)

y = np.zeros((T, 2))

precision = np.zeros((T, 2, 2))               # precision for Gaussian distribution

precision_inverse = np.zeros((T, 2, 2))

s[0] = 1

x[0] = 0

n = 1

precision[0] = wishart.rvs(scale=V0, df=n0)

precision_inverse[0] = np.linalg.inv(precision[0])

y[0] = multivariate_normal.rvs(mean=(0, 0), cov=precision_inverse[0])

for t in range(1, T):

    p = alpha/(n+alpha)

    s[t] = bernoulli.rvs(p)

    x[t] = s[t] + x[t-1]

    if s[t] == 0:
        n = n + 1
        precision[t] = precision[t-1]
        precision_inverse[t] = precision_inverse[t-1]
    else:
        n = 1
        precision[t] = wishart.rvs(scale=V0, df=n0)
        precision_inverse[t] = np.linalg.inv(precision[t])

    y[t] = multivariate_normal.rvs(mean=(0, 0), cov=precision_inverse[t])


# ---------------------------------------------------------------------------------------
#
#  infer the signals using Gibbs sampling algorithm
#
# ---------------------------------------------------------------------------------------

# ---------------------- #
#  1. initialization     #
# ---------------------- #

V0_e = np.matrix([[1, 0], [0, 1]])
n0_e = 3
V_e = np.matrix([[1, 0], [0, 1]])                   # Wishart parameter matrix
nu_e = 3
a_e = 1
b_e = 1
D = 2
alpha_e = 0.75

x_e = np.zeros(T)
precision_e = np.zeros((T, 2, 2))
precision_e_inverse = np.zeros((T, 2, 2))

for i in range(0, len(x_e)):
    if i%3 == 0:            # new regime
        if i == 0:
            x_e[i] = 0
        else:
            x_e[i] = x_e[i-1] + 1
        precision_e[i] = wishart.rvs(scale=V_e, df=nu_e+D-1)
        precision_e_inverse[i] = np.linalg.inv(precision_e[i])
    else:                   # keep the same regime
        x_e[i] = x_e[i-1]
        precision_e[i] = precision_e[i-1]
        precision_e_inverse[i] = precision_e_inverse[i-1]

n_e = book_keeping_n(x_e)

# ----------------------- #
#  2. inference           #
# ----------------------- #

inf_rep = 10000

for inf_rep_count in range(0, inf_rep):

    print(str(inf_rep_count/inf_rep*100)+"%")

    # sample x

    for t in range(0, T):

        if t == 0 and x_e[t] == x_e[t+1]:

            # case 1
            continue

        elif t == 0 and x_e[t] != x_e[t+1]:

            # case 2
            continue

        elif t != 0 and t != T-1 and x_e[t] != x_e[t-1] and x_e[t] == x_e[t+1]:

            # case 3
            p1 = (n_e[int(x_e[t])] - 1)/(n_e[int(x_e[t])] + alpha_e) \
                 * multivariate_normal.pdf(y[t], mean=(0, 0), cov=precision_e_inverse[t])
            p2 = n_e[int(x_e[t-1])]/(n_e[int(x_e[t-1])] + alpha_e + 1) \
                 * multivariate_normal.pdf(y[t], mean=(0, 0), cov=precision_e_inverse[t-1])
            p3 = alpha_e/(1 + alpha_e) * multi_student_pdf(y[t], df=nu_e, loc=(0,0), scale=nu_e*V_e)

            sum_temp = p1+p2+p3

            x_e[t] = np.random.choice([x_e[t], x_e[t-1], -10], size=1, p=[p1/sum_temp, p2/sum_temp, p3/sum_temp])

            x_e = book_keeping_z(x_e)
            n_e = book_keeping_n(x_e)

            continue

        elif t != 0 and t != T-1 and x_e[t] != x_e[t-1] and x_e[t] != x_e[t+1]:

            # case 4
            p1 = (n_e[int(x_e[t]-1)])/(n_e[int(x_e[t]-1)] + alpha_e + 1) \
                 * multivariate_normal.pdf(y[t-1], mean=(0,0), cov=precision_e_inverse[t-1])
            p2 = (n_e[int(x_e[t]+1)])/(n_e[int(x_e[t]+1)] + alpha_e + 1) \
                 * multivariate_normal.pdf(y[t+1], mean=(0,0), cov=precision_e_inverse[t+1])
            p3 = alpha_e/(1 + alpha_e) * multi_student_pdf(y[t], df=nu_e, loc=0, scale=nu_e*V_e)

            sum_temp = p1 + p2 + p3

            x_e[t] = np.random.choice([x_e[t-1], x_e[t+1], x_e[t]], size=1,
                                      p=[p1 / sum_temp, p2 / sum_temp, p3 / sum_temp])

            x_e = book_keeping_z(x_e)
            n_e = book_keeping_n(x_e)

            continue

        elif t != 0 and t != T-1 and x_e[t] == x_e[t-1] and x_e[t] != x_e[t+1]:

            # case 5
            p1 = (n_e[int(x_e[t])]-1) / (n_e[int(x_e[t])] + alpha_e) \
                 * multivariate_normal.pdf(y[t], mean=(0,0), cov=precision_e_inverse[t])
            p2 = (n_e[int(x_e[t] + 1)]) / (n_e[int(x_e[t] + 1)] + alpha_e + 1) \
                 * multivariate_normal.pdf(y[t + 1], mean=(0,0), cov=precision_e_inverse[t+1])
            p3 = alpha_e / (1 + alpha_e) * multi_student_pdf(y[t], df=nu_e, loc=0, scale=nu_e * V_e)

            sum_temp = p1 + p2 + p3

            x_e[t] = np.random.choice([x_e[t - 1], x_e[t + 1], -10], size=1,
                                      p=[p1 / sum_temp, p2 / sum_temp, p3 / sum_temp])

            x_e = book_keeping_z(x_e)
            n_e = book_keeping_n(x_e)

            continue

        elif t == T-1 and x_e[t] == x_e[t-1]:
            # case 6
            continue

        elif t == T-1 and x_e[t] != x_e[t-1]:
            # case 7
            continue
        else:
            # any other cases
            continue

    # sample precision matrix

    for regime_count in range(0, int(x_e[-1]+1)):

        n_star = n_e[regime_count] + nu_e + D - 1

        V_e_inv = np.linalg.inv(V_e)

        Y_e = np.zeros((int(n_e[regime_count]),2))

        n_e_count = 0

        flag_x = 0

        for x_index in range(0,len(x_e)):
            if x_e[x_index] == regime_count:
                flag_x = 1
                Y_e[n_e_count][0] = y[x_index][0]
                Y_e[n_e_count][1] = y[x_index][1]
                n_e_count = n_e_count + 1
            elif flag_x == 1:
                break

        Y_e_trans = np.matrix.transpose(Y_e)

        Y_sq = np.matmul(Y_e_trans, Y_e)

        V_star_inv = V_e_inv + Y_sq

        V_e_rand = wishart.rvs(df=n_star, scale=np.linalg.inv(V_star_inv))

        V_e_rand_inv = np.linalg.inv(V_e_rand)

        for regime_count_x in range(0,len(x_e)):
            if x_e[regime_count_x] == regime_count:

                precision_e[regime_count_x] = V_e_rand
                precision_e_inverse[regime_count_x] = V_e_rand_inv

    # sample V

    V_L = V0_e

    n_L = n0_e + (x_e[-1] + 1) * (D + nu_e - 1)

    for regime_count in range(0, int(x_e[-1]+1)):

        for x_index in range(0,len(x_e)):

            if x_e[x_index] == regime_count:

                V_L = V_L + precision_e[int(x_index)]

                break

    V_e = invwishart.rvs(df=n_L, scale=V_L)

    # sample alpha

    w = 0

    for i in n_e:
        w_i = beta.rvs(a=alpha_e+1, b=i)
        w = w + log(w_i)

    alpha_e = sci_gamma.rvs(a=a_e+x_e[-1]+1, scale=b_e-w)


N_vector = range(1,T+1)

cov_trace = np.zeros(T)
cov_true = np.zeros(T)

for i in range(0,T):

    cov_trace[i] = precision_e_inverse[i][0][1]
    cov_true[i] = precision_inverse[i][0][1]

fig1, ax1 = plt.subplots()
ax1.plot(N_vector, cov_trace, label="est_covariance")
ax1.plot(N_vector, cov_true, label="true_covariance")
ax1.legend(fontsize=14)
plt.savefig("bi-variate_covariance.pdf")
