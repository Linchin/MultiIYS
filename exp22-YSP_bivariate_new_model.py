__author__ = "Lingqing Gan"

"""
File Name: 
exp22-YSP_bivariate_new_model.py

Note 01/15/2020 (exp22)
At first we were thinking about a linear regression for a,
but later prof. came up with a Bayesian method integrated
with Gibbs sampling. Now we alter the code into this form.

To-Dos:
* priors
* change gibbs
* len(a) should be adjustable
* fix the Gibbs
    - mean variance (w/o burn-in)
    - mean alpha (w/o burn-in)

x[t] = a0 + a1*x[t-1] + a2*x[t-2] + u[t]

Notes 01/13/2020 (exp22)
Now we have confirmed that exp21 is a working version of the 
regime/partition detection program. Now we change it so we
can add more complexity to the model.

Notes 01/13/2020 (exp21)
Before moving on to the new model, let's adapt and understand the 
code, so we have a working version. Changed the file name to 
exp21-YSP_bivariate_updated.py

This version is called bivariate, and from what I understand
from the code, this is still a single node model, where we apply
the original algorithm. What is updated is, instead of a 1-d random
signal, we receive a 2-d random signal with a randomly generated
covariance matrix under a Wishart distribution.

This is natural because Wishart distribution is a multi-D expansion
of Gamma distribution.

Notes 01/10/2020
Based on the original YSP_bi-variate.py, we change the model
slightly so the random signal is not generated from a Gaussian 
process directly.

x[t] = a1*x[t-1] + a2*x[t-2] + u[t]

we use linear fitting to estimate a1 and a2.
Then we use the same Gibbs sampling for the estimation, just the old
way.

Let's first check out how we did it in the original 2018 code.

"""


# ------------------------------------------------------------------------------------

#   FILE_NAME:      YSP_bi-variate.py

#   DESCRIPTION:    Gibbs sampling for bi-variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 6.

#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           07/01/2018

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
    """
    turn the z sequence into a sequence of the length of each regime
    :param z: the sequence of regime indices
    :return: a sequence of the length of each regime
    """

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
    """
    Reorganize the values of z sequence after a new estimation.
    :param z:  the sequence of regime indices
    :return: the updated z sequence
    """

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
        d: the density of the given element
    input:
        x: parameter (d dimensional numpy array or scalar)
        mu: mean (d dimensional numpy array or scalar)
        Sigma: scale matrix (dxd numpy array)
        df: degrees of freedom
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

a = 1                                           # the YS parameter
b = 1
alpha = np.random.gamma(a, scale=1/b)           # shape, scale, b is rate


T = 100                                        # total number of time instants

V0 = np.array([[1, 0], [0, 1]])

n0 = 3

v = 2                                           # degree of freedom

a_min = -1                          # assume each element of vector
a_max = 1                           # a is uniform dist on [a_min, a_max]

# ---------------------------------------------------------------------------------------
#
# Generative Model:
#
# generate random signals according to Yule-Simon process
#
# --------------------------------------------------------------------

s = np.zeros(T)                 # the index of the current regime

x = np.zeros(T)                 # new regime indicator

y = np.zeros((T, 2))            # the random Gaussian signals

precision = np.zeros((T, 2, 2))  # precision for Gaussian random signals

precision_inverse = np.zeros((T, 2, 2))

li_coef_length = 2              # the length of the coef vector a

a = np.zeros((T, li_coef_length))         # the linear coefficient vector a

signal_mean_vector = np.zeros((T, 2))   # the sequence that saves the mean value of the signals

s[0] = 1

x[0] = 0

n = 1                   # counter for the length of the current regime

precision[0] = wishart.rvs(scale=V0, df=n0)

precision_inverse[0] = np.linalg.inv(precision[0])

y[0] = multivariate_normal.rvs(mean=(0, 0), cov=precision_inverse[0])

# generate the random signal

for t in range(1, T):

    p = alpha/(n+alpha)
    s[t] = bernoulli.rvs(p)
    x[t] = s[t] + x[t-1]

    if s[t] == 0:
        # remain in the same regime
        n = n + 1
        precision[t] = precision[t-1]
        precision_inverse[t] = precision_inverse[t-1]
        for a_index in range(li_coef_length):
            a[t, a_index] = a[t-1, a_index]

    else:
        # start a new regime
        n = 1
        precision[t] = wishart.rvs(scale=V0, df=n0)
        precision_inverse[t] = np.linalg.inv(precision[t])
        for a_index in range(li_coef_length):
            a[t, a_index] = np.random.uniform(low=a_min,
                                              high=a_max)

    # calculate the mean values of the new signal
    signal_mean_vector[t, 0] = a[t, 0] + a[t, 1] * signal_mean_vector[t-1, 0]
    signal_mean_vector[t, 1] = a[t, 0] + a[t, 1] * signal_mean_vector[t-1, 1]

    # generate the new signals based on the mean values and covariance matrices
    y[t] = multivariate_normal.rvs(mean=signal_mean_vector[t],
                                   cov=precision_inverse[t])


# ---------------------------------------------------------------------------------------
#
#  infer the signals using Gibbs sampling algorithm
#
# ---------------------------------------------------------------------------------------

# ---------------------- #
#  1. initialization     #
# ---------------------- #

# add process and hyperparameter to generate the linear parameters

V0_e = np.array([[1, 0], [0, 1]])   # starter parameter for Gibbs
n0_e = 3                            # hyperparameter of degree of freedom
V_e = np.array([[1, 0], [0, 1]])    # Wishart parameter matrix
nu_e = 3                            # Wishart hyperparameter used for partitioning, degree of freedom

alpha_a_e = 1                             # starter parameter for Gibbs
alpha_b_e = 1                             # starter parameter for Gibbs

D = 2                               # (Hyper) degree of freedom for the Wishart dist for the precision matrix

alpha_e = 0.75                      # (Gibbs) alpha value

li_coef_length = 2                  # number of the linear coefficients, first just 2. constant and 1st order
a_e = np.zeros((T, li_coef_length)) # save the data for a coefficient for each time instant


# y[t] is the original signal
# y_adj[t] is the mean-adjusted signal
y_adj = np.zeros((T, 2, 2))

x_e = np.zeros(T)                           # regime partitions
precision_e = np.zeros((T, 2, 2))           # (Gibbs) precision matrix for each time instant
precision_e_inverse = np.zeros((T, 2, 2))   # (Gibbs) precision matrix inverse

# ---------------------------------
#       Gibbs Initialization
# ---------------------------------
# create an arbitrary estimation as the Gibbs starter
# parameters estimated:
# a vector
# partitions
# precision matrix for each partition
for i in range(0, len(x_e)):
    if i % 3 == 0:
        # arbitrarily decide where the new regimes start, we use every
        # three time slots
        if i == 0:
            x_e[i] = 0
        else:
            x_e[i] = x_e[i-1] + 1
        # generate the new precision matrix/covariance matrix
        # for the new regime
        precision_e[i] = wishart.rvs(scale=V_e, df=nu_e+D-1)
        precision_e_inverse[i] = np.linalg.inv(precision_e[i])
        # generate the linear coefficients for this regime
        for a_index in range(li_coef_length):
            a_e[i, a_index] = np.random.uniform(low=a_min,
                                                high=a_max)
    else:
        # if we remain in the same regime, we use the same
        # precision mtx/covariance mtx, and the new linear
        # coefficients
        x_e[i] = x_e[i-1]
        precision_e[i] = precision_e[i-1]
        precision_e_inverse[i] = precision_e_inverse[i-1]
        for a_index in range(li_coef_length):
            a_e[i, a_index] = a_e[i-1, a_index]

# generate the initial y_adj
mean_sequence = np.zeros((T, 2))
for t in range(1, T):
    # generate the mean sequence
    mean_sequence[t, 0] = a_e[t, 0] + a_e[t, 1] * mean_sequence[t-1, 0]
    mean_sequence[t, 1] = a_e[t, 0] + a_e[t, 1] * mean_sequence[t-1, 1]

# calculate the mean-adjusted signal
y_adj = np.subtract(y, mean_sequence)

n_e = book_keeping_n(x_e)

# ----------------------- #
#  2. inference           #
# ----------------------- #

# (need to add burn-in)
inf_rep = 200              # Gibbs sampling repetitions (??)

for inf_rep_count in range(0, inf_rep):

    (str(inf_rep_count/inf_rep*100)+"%")

    # sample x

    """
    # decide the partitions
    # how to determine the mean value?
    # how do we save the mean value?
    # maybe we need to add a mean value bookkeeping :)
    # add an estimated baseline
    # add a section to estimate the baseline parameter
    # start from one parameter.
    """

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
                * multivariate_normal.pdf(y_adj[t], mean=(0, 0), cov=precision_e_inverse[t])
            p2 = n_e[int(x_e[t-1])]/(n_e[int(x_e[t-1])] + alpha_e + 1) \
                * multivariate_normal.pdf(y_adj[t], mean=(0, 0), cov=precision_e_inverse[t-1])
            p3 = alpha_e/(1 + alpha_e) * multi_student_pdf(y_adj[t], df=nu_e, loc=(0,0), scale=nu_e*V_e)

            sum_temp = p1+p2+p3

            x_e[t] = np.random.choice([x_e[t], x_e[t-1], -10],
                                      size=1,
                                      p=[p1/sum_temp, p2/sum_temp, p3/sum_temp])

            x_e = book_keeping_z(x_e)
            n_e = book_keeping_n(x_e)

            continue

        elif t != 0 and t != T-1 and x_e[t] != x_e[t-1] and x_e[t] != x_e[t+1]:

            # case 4
            p1 = (n_e[int(x_e[t]-1)])/(n_e[int(x_e[t]-1)] + alpha_e + 1) \
                * multivariate_normal.pdf(y_adj[t-1], mean=(0, 0), cov=precision_e_inverse[t-1])
            p2 = (n_e[int(x_e[t]+1)])/(n_e[int(x_e[t]+1)] + alpha_e + 1) \
                * multivariate_normal.pdf(y_adj[t+1], mean=(0, 0), cov=precision_e_inverse[t+1])
            p3 = alpha_e/(1 + alpha_e) * multi_student_pdf(y_adj[t], df=nu_e, loc=0, scale=nu_e*V_e)

            sum_temp = p1 + p2 + p3

            x_e[t] = np.random.choice([x_e[t-1], x_e[t+1], x_e[t]], size=1,
                                      p=[p1 / sum_temp, p2 / sum_temp, p3 / sum_temp])

            x_e = book_keeping_z(x_e)
            n_e = book_keeping_n(x_e)

            continue

        elif t != 0 and t != T-1 and x_e[t] == x_e[t-1] and x_e[t] != x_e[t+1]:

            # case 5
            p1 = (n_e[int(x_e[t])]-1) / (n_e[int(x_e[t])] + alpha_e) \
                * multivariate_normal.pdf(y_adj[t], mean=(0, 0), cov=precision_e_inverse[t])
            p2 = (n_e[int(x_e[t] + 1)]) / (n_e[int(x_e[t] + 1)] + alpha_e + 1) \
                * multivariate_normal.pdf(y_adj[t + 1], mean=(0, 0), cov=precision_e_inverse[t+1])
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

    # sample a vector

    for regime_count in range(0, int(x_e[-1] + 1)):
        # for each regime:

        # length of current regime
        current_length = n_e[regime_count]

        # patch: skip current regime if length is 1
        if current_length == 1:
            # save the a values to the a matrix within this current regime
            for x_index in range(0, len(x_e)):
                if x_e[x_index] == regime_count:
                    a_e[x_index, :] = a[x_index, :]
                    break
            continue

        # find signals of the current regime
        # (just use the first dimension of the observed signals)
        cur_reg_signals = []
        for x_index in range(0,len(x_e)):
            if x_e[x_index] == regime_count:
                cur_reg_signals.append(y[x_index][0])

        cur_reg_signals = np.array(cur_reg_signals)

        # the estimated precision and covariance of the current regime
        # signals
        for x_index in range(0,len(x_e)):
            if x_e[x_index] == regime_count:
                current_regime_sig_precision = precision_e[x_index]
                current_regime_sig_covariance = precision_e_inverse[x_index]
                break

        x0_precision = current_regime_sig_precision[0][0]
        x0_variance = current_regime_sig_covariance[0][0]

        # construct the H matrix
        # (This is the H matrix for a 2D signal)
        # changed back
        H_mat_height = int(current_length)
        H_mat_width = li_coef_length
        H_mat = np.zeros((H_mat_height, H_mat_width))

        for local_t in range(H_mat_height):
            for coef_index in range(0, li_coef_length):
                if coef_index == 0:
                    H_mat[local_t][coef_index] = 1
                    # H_mat[local_t + current_length][coef_index] = 1
                elif coef_index <= local_t:      # we just use the first dimension here
                    H_mat[local_t][coef_index] = cur_reg_signals[local_t]
                    # H_mat[local_t + current_length][coef_index] = cur_reg_signals[local_t][1]

        # calculate mean vector
        square_mat = np.matmul(H_mat.transpose(), H_mat)
        square_mat_inverse = np.linalg.inv(square_mat)
        mult_temp = np.matmul(square_mat_inverse, H_mat.transpose())
        a_mean = np.matmul(mult_temp, cur_reg_signals)

        # calculate covariance matrix
        a_precision = x0_precision**2 * square_mat
        a_covariance = np.linalg.inv(a_precision)

        # generate random a vector
        a_random = np.random.multivariate_normal(mean=a_mean,
                                                 cov=a_covariance)

        # save the a values to the a matrix within this current regime
        for x_index in range(0, len(x_e)):
            if x_e[x_index] == regime_count:
                a_e[x_index, :] = a_random[:]

    # calculate the mean value sequence
    mean_sequence = np.zeros((T, 2))
    for t in range(1, T):
        mean_sequence[t][0] = a_e[t][0] + a_e[t][1] * mean_sequence[t-1][0]
        mean_sequence[t][1] = a_e[t][0] + a_e[t][1] * mean_sequence[t-1][1]

    # calculate the mean-adjusted signal
    y_adj = np.subtract(y, mean_sequence)

    # %% sample precision matrix

    for regime_count in range(0, int(x_e[-1]+1)):
        # for each regime, sample the precision matrix
        # of this regime

        # for a multi-variate normal dist, the precision matrix
        # is Wishart dist

        # n_star is the a posterior degree of freedom
        n_star = n_e[regime_count] + nu_e + D - 1

        V_e_inv = np.linalg.inv(V_e)

        # use Y_e to convert prior to posterior
        Y_e = np.zeros((int(n_e[regime_count]), 2))

        n_e_count = 0

        flag_x = 0

        # what is this used for???
        # changed this part
        for x_index in range(0, len(x_e)):
            if x_e[x_index] == regime_count:
                Y_e[n_e_count][0] = y_adj[x_index][0]
                Y_e[n_e_count][1] = y_adj[x_index][1]
                n_e_count = n_e_count + 1
                if n_e_count >= len(Y_e):
                    break

        Y_e_trans = np.transpose(Y_e)

        Y_sq = np.matmul(Y_e_trans, Y_e)

        # scale matrix of the a posterior Wishart distribution
        V_star_inv = V_e_inv + Y_sq

        # a posterior of the precision matrix is Wishart
        V_e_rand = wishart.rvs(df=n_star, scale=np.linalg.inv(V_star_inv))

        V_e_rand_inv = np.linalg.inv(V_e_rand)

        # save the estimated precision matrix to the time series
        for regime_count_x in range(0, len(x_e)):
            if x_e[regime_count_x] == regime_count:
                precision_e[regime_count_x] = V_e_rand
                precision_e_inverse[regime_count_x] = V_e_rand_inv

    # sample V
    # new value is V_e
    # V is the hyperparameter of precision matrix
    # Conjugate prior for the covariance matrix
    # inverse-Wishart prior, Multivariate normal likelihood

    # scale matrix
    V_L = V0_e

    # degree of freedom
    n_L = n0_e + (x_e[-1] + 1) * (D + nu_e - 1)

    # update the scale matrix
    for regime_count in range(0, int(x_e[-1]+1)):
        # for each regime

        for x_index in range(0, len(x_e)):
            if x_e[x_index] == regime_count:
                V_L = V_L + precision_e[int(x_index)]
                break

    V_e = invwishart.rvs(df=n_L, scale=V_L)

    # sample alpha
    w = 0
    for i in n_e:
        w_i = beta.rvs(a=alpha_e+1, b=i)
        w = w + log(w_i)
    alpha_e = sci_gamma.rvs(a=alpha_a_e+x_e[-1]+1, scale=alpha_b_e-w)


N_vector = range(1, T+1)
cov_trace = np.zeros(T)
cov_true = np.zeros(T)

for i in range(0, T):
    cov_trace[i] = precision_e_inverse[i][0][1]
    cov_true[i] = precision_inverse[i][0][1]

fig1, ax1 = plt.subplots()
ax1.plot(N_vector, cov_trace, label="est_covariance")
ax1.plot(N_vector, cov_true, label="true_covariance")
ax1.plot(N_vector, a[:, 0], label="true_a0")
ax1.plot(N_vector, a[:, 1], label="true_a1")
ax1.plot(N_vector, a_e[:, 0], label="est_a0")
ax1.plot(N_vector, a_e[:, 1], label="est_a1")


ax1.legend(fontsize=14)
plt.show()
# plt.savefig("bi-variate_covariance.pdf")
