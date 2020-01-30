__author__ = "Lingqing Gan"

"""
File Name: 
exp25-YSP_movariate.py

Notes 01/28/2020:
This is the original code - single variable version.
This is the one that works.
Now the task is to convert it to a more up to date version 
and capable of saving the data. Then we add the estimation of 
vector a.

Add functions to save data.
"""


# ------------------------------------------------------------------------------------

#   FILE_NAME:      YSP_movariate.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           06/19/2018 - 06/29/2018

# ------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t as student
from math import sqrt
from math import log


# packages used to save the data

import time
import pickle
import os
from os.path import join

time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

# data section of the dict to be saved
data_dict = {}
z_e_record = {}
v_e_record = {}

# ---------------------------------------------------------------------------------------
#
#  function definition and invocation
#
# ---------------------------------------------------------------------------------------


def book_keeping_n(z):

    n = np.array([])
    for i in range(0,len(z)):
        if i == 0:
            n_count = 1
            temp = z[i]
            continue
        elif i < len(z) - 1:
            if z[i] == temp:
                n_count = n_count + 1
            else:
                n = np.append(n,n_count)
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

    flag3 = 0

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

            for i in range(new+1,len(z)):
                z[i] = z[i] + 1

    if flag2 == 1 and flag1 == 0:
        for i in range(gap, len(z)):
            z[i] = z[i]-1

    for i in range(0,len(z)-1):
        if z[i+1] > z[i] + 1:
            print("error")
            break

    return z


# hyper parameters

a = 1
b = 1
c = 1
d = 1

T = 300              # length of time series


# ---------------------------------------------------------------------------------------
#
#  generate random signals according to Yule-Simon process
#
# ---------------------------------------------------------------------------------------


# alpha = np.random.gamma(shape=a,scale=1/b)        # calibration parameter

alpha = 0.75

# variables

s = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
z = np.zeros(T)     # regime index at [t].
v = np.zeros(T)     # signal precision (lambda) at [t]
x = np.zeros(T)     # signal at [t]


s[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1

z[0] = 1            # the first node is automatically assigned to the 1st regime

v[0] = np.random.gamma(shape=c, scale=1/d)     # assign the first precision

x[0] = np.random.normal(loc=0, scale=sqrt(1/v[0]))       # signal

n_count = 1         # number of nodes in the current regime


for t in range(1, T):

    # update value of p (probability of creating new regime)
    p = alpha / (n_count + alpha)

    # if s[t] = 1, then x[t] belongs to a new regime
    s[t] = np.random.binomial(1, p)

    # repeat:
    # n_count: update the number of nodes in the current regime
    # z:       assign regime number to the t'th node
    # v:       assign precision
    # x:       assign i.i.d. signal
    if s[t] == 1:
        n_count = 1
        z[t] = z[t - 1] + 1
        v[t] = np.random.gamma(shape=c, scale=1/d)
        x[t] = np.random.normal(loc=0, scale=sqrt(1/v[t]))

    else:
        n_count = n_count + 1
        z[t] = z[t - 1]
        v[t] = v[t - 1]
        x[t] = np.random.normal(loc=0, scale=sqrt(1 / v[t]))


# ---------------------------------------------------------------------------------------
#
#  infer the signals using Gibbs sampling algorithm
#
# ---------------------------------------------------------------------------------------


# ---------------------- #
#  1. initialization     #
# ---------------------- #

# initial estimation values of parameters
alpha_e = 1
a_e = 1
b_e = 1
c_e = 1
d_e = 1

z_e = np.zeros(T)     # inference: indicator of regime index at [t].
v_e = np.zeros(T)     # inference: indicator of variance at [t]
s_e = np.zeros(T)     # inference: indicator of new regime

n_count_e = 1         # inference: number of nodes in the current regime

z_e[0] = 0            # (with inference:) the first node is automatically assigned to the 0st regime
s_e[0] = 1

c_e_v = 3/2                       # the values of c_i and d_i are updated on themselves, replacing the previous values

d_e_v = 1 + 0.5 * x[0] ** 2       # c_i and d_i are Gamma parameters to draw precision

v_e[0] = np.random.gamma(shape=c_e, scale=1/d_e)       # draw precision

for t in range(1, T):

    # calculate corresponding prob

    p1 = n_count_e / (n_count_e + alpha_e) * norm.pdf(x[t],
                                                      loc=0,
                                                      scale=sqrt(1/v_e[t-1]))

    p2 = alpha_e / (n_count_e + alpha_e) * student.pdf(x[t],
                                                       2*c_e,
                                                       loc=0,
                                                       scale=sqrt(d_e/c_e))

    p_temp = p2/(p1+p2)                 # p_temp is the probability that there's a new regime

    # roll the Bernoulli dice
    s_e[t] = np.random.binomial(1, p_temp)      # s_e[t]=1 means new regime

    if s_e[t] == 1:

        # new regime
        # regime index & count
        z_e[t] = z_e[t-1] + 1
        n_count_e = 1

        # draw precision from Gamma distribution
        c_e_v = 3/2
        d_e_v = 1 + 0.5 * x[t] ** 2
        v_e[t] = np.random.gamma(shape=c_e_v, scale=1/d_e_v)

    else:

        # not new regime
        # regime index & count
        z_e[t] = z_e[t-1]
        n_count_e = n_count_e + 1

        # draw precision from Gamma distribution
        c_e_v = c_e_v + 1/2
        d_e_v = d_e_v + 1/2 * x[t] ** 2
        v_e[(t-n_count_e+1):(t+1)] = np.random.gamma(shape=c_e_v, scale=1/d_e_v)


# ---------------------- #
#  2.    inference       #
# ---------------------- #

rep = 1000           # rounds of Gibbs sampling
rep_alpha = 1000     # rounds of Gibbs sampler for alpha

alpha_estimate = np.zeros(rep_alpha)

regime_count_e = np.zeros(rep)

alpha_e_record = np.zeros(rep)


n_i = book_keeping_n(z_e)

for rep_index in range(0, rep):

    print(rep_index)

    # draw z_t
    # (only consider the 2nd till and last node)
    for t in range(1,T-1):

        if z_e[t] == z_e[t-1] and z_e[t] == z_e[t+1]:

            # no boundary then pass
            continue

        elif z_e[t] == z_e[t-1] and z_e[t] != z_e[t+1]:

            # right boundary

            # calculate the probs

            # merge with left (which is equal to remaining the same)
            p_z_1 = (n_i[int(z_e[t])]-1)/(
                    n_i[int(z_e[t])] + alpha_e
                    ) * norm.pdf(x[t],
                                 loc=0,
                                 scale=sqrt(1/v_e[t]))

            # merge with right
            p_z_2 = (n_i[int(z_e[t+1])])/(
                    n_i[int(z_e[t+1])] + alpha_e + 1
                    ) * norm.pdf(x[t],
                                 loc=0,
                                 scale=sqrt(1/v_e[t+1]))

            # create new regime
            p_z_k = alpha_e / (alpha_e + 1) * student.pdf(x[t],
                                                          2*c_e,
                                                          loc=0,
                                                          scale=sqrt(d_e/c_e))

            sum = p_z_1 + p_z_2 + p_z_k

            # draw
            z_e[t] = np.random.choice([z_e[t], z_e[t+1], -10],
                                      1,
                                      p=[p_z_1/sum, p_z_2/sum, p_z_k/sum])

            z_e = book_keeping_z(z_e)
            n_i = book_keeping_n(z_e)

            continue

        elif z_e[t] != z_e[t-1] and z_e[t] == z_e[t+1]:

            # left boundary

            # calculate the probs

            # merge with right (which is equal to remaining the same)
            p_z_1 = (n_i[int(z_e[t])]-1)/(
                    n_i[int(z_e[t])] + alpha_e
                    ) * norm.pdf(x[t],
                                 loc=0,
                                 scale=sqrt(1/v_e[t]))

            # merge with left
            p_z_2 = (n_i[int(z_e[t]-1)])/(
                    n_i[int(z_e[t]-1)] + alpha_e + 1
                    ) * norm.pdf(x[t],
                                 loc=0,
                                 scale=sqrt(1/v_e[t-1]))

            # create new regime
            p_z_k = alpha_e / (alpha_e + 1
                               ) * student.pdf(x[t],
                                               2*c_e,
                                               loc=0,
                                               scale=sqrt(d_e/c_e))

            sum_temp = p_z_1 + p_z_2 + p_z_k

            # draw
            z_e[t] = np.random.choice([z_e[t], z_e[t-1], -10],
                                      1,
                                      p=[p_z_1/sum_temp,
                                         p_z_2/sum_temp,
                                         p_z_k/sum_temp])

            z_e = book_keeping_z(z_e)
            n_i = book_keeping_n(z_e)

            continue

        elif z_e[t] != z_e[t - 1] and z_e[t] != z_e[t + 1]:

            # double boundary

            # calculate the probs

            # merge with left
            p_z_1 = (n_i[int(z_e[t]-1)]) / (
                    n_i[int(z_e[t]-1)] + alpha_e + 1
                    ) * norm.pdf(x[t], loc=0, scale=sqrt(1 / v_e[t-1]))

            # merge with right
            p_z_2 = (n_i[int(z_e[t] + 1)]) / (
                    n_i[int(z_e[t] + 1)] + alpha_e + 1
                    ) * norm.pdf(x[t], loc=0, scale=sqrt(1 / v_e[t + 1]))

            # create new regime (which means remaining the same)
            p_z_k = alpha_e / (alpha_e + 1) * student.pdf(
                x[t], 2 * c_e, loc=0, scale=sqrt(d_e / c_e))

            sum_temp = p_z_1 + p_z_2 + p_z_k

            # draw
            z_e[t] = np.random.choice([z_e[t - 1], z_e[t + 1], z_e[t]],
                                      1,
                                      p=[p_z_1/sum_temp,
                                         p_z_2/sum_temp,
                                         p_z_k/sum_temp])

            z_e = book_keeping_z(z_e)
            n_i = book_keeping_n(z_e)

            continue

    regime_count_e[rep_index] = z_e[-1] + 1

    # draw alpha Gibbs sampler
    for rep_alpha_index in range(0, rep_alpha):

        # draw w_j
        # draw alpha

        b_draw_alpha = b_e

        for i in range(0, len(n_i)):

            w = np.random.beta(a=alpha_e+1,
                               b=n_i[i],
                               size=1)

            b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n_i)

        alpha_e = np.random.gamma(shape=a_draw_alpha,
                                  scale=1 / b_draw_alpha)

        alpha_estimate[rep_alpha_index] = alpha_e

    alpha_e_record[rep_index] = alpha_e

    # draw v_e[0:T-1]
    for i in range(0, len(n_i)):

        d_e_v = d_e

        c_e_v = c_e + n_i[i] / 2

        for j in range(0, len(z_e)):

            if z_e[j] == i:

                d_e_v = d_e_v + 1/2 * x[j] ** 2

        v_i = np.random.gamma(shape=c_e_v,
                              scale=1/d_e_v)

        for j in range(0, len(z_e)):

            if z_e[j] == i:

                v_e[j] = v_i

    z_e_record[rep_index] = z_e.copy()
    print(z_e)
    print(z_e_record[rep_index])
    v_e_record[rep_index] = v_e.copy()

# ---------------------------------------------------------------------------------------
#
#  plot the data
#
# ---------------------------------------------------------------------------------------

N_vector = range(1,T+1)

fig1, ax1 = plt.subplots()


ax1.plot(N_vector, v, label="v[T]")
# ax1.plot(N_vector, x, label="x[T]")
ax1.plot(N_vector, v_e, label="v_e[T]")

ax1.legend(fontsize=14)

plt.savefig("exp25-data-" + time_string + "-precision_compare.pdf")

# draw regime count

rep_vector = range(1, rep+1)

fig2, ax2 = plt.subplots()

regime_count = np.repeat(z[-1]+1,rep)

ax2.plot(rep_vector, regime_count, label="real regime")
ax2.plot(rep_vector, regime_count_e, label="z_e[-1]")

ax2.legend(fontsize=14)

plt.savefig("exp25-data-" + time_string + "-regime_count.pdf")


# draw alpha value comparison

fig3, ax3 = plt.subplots()

alpha_true_vector = np.repeat(alpha, rep)

ax3.plot(rep_vector, alpha_true_vector, label="real alpha")
ax3.plot(rep_vector, alpha_e_record, label="alpha_e_vector")

ax3.legend(fontsize=14)

plt.savefig("exp25-data-" + time_string + "-alpha_trace.pdf")

# draw signal x vs. std

fig4, ax4 = plt.subplots()

std_vector = [1/sqrt(i) for i in v]

ax4.plot(N_vector, x, label="x")
ax4.plot(N_vector, std_vector, label="std")

ax4.legend(fontsize=14)

plt.savefig("exp25-data-" + time_string + "-x_vs_std.pdf")

# 01/28/2020 addition
# plot the regime partitions
fig5, ax5 = plt.subplots()
ax5.plot(N_vector, z, label="true partition")
ax5.plot(N_vector, z_e, label="estimated partition")

ax5.legend(fontsize=14)

plt.savefig("exp25-data-" + time_string + "-partition_compare.pdf")

# ---------------------------------------------------------------------------------------
#
#  save the data
#
# ---------------------------------------------------------------------------------------

# organize the data sub dict to be saved
data_dict["true_regimes"] = z
data_dict["true_signals"] = x
data_dict["true_precision"] = v

data_dict["estimated_regimes"] = z_e_record
data_dict["estimated_precision_mtx"] = v_e_record
data_dict["estimated_alpha"] = alpha_e_record


# create the dict to save
# parameter section + data section
save_dict = {"parameters": {"Gibbs sampling iterations": rep,
                            "alpha Gibbs sampling iterations": rep_alpha,
                            "signal_dimension": 1,
                            "alpha true value": alpha,
                            "Sequence_length": T
                            },
             "data": data_dict}

# absolute dir the script is in
script_dir = os.path.dirname(__file__)
rel_path_temp = "result_temp"

# the file name
file_name = "exp25-data-" + time_string + "(YSP_single_variable).pickle"
complete_file_name = join(script_dir, rel_path_temp, file_name)
print("Saved file name: ", file_name)

# save the file
with open(complete_file_name, 'wb') as handle:
    pickle.dump(save_dict, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
    print("Data saved successfully!")