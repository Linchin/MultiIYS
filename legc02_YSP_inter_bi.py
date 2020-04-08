__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      legc02_YSP_inter_bi.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.

#                   Adding: two variables where one is influencing another


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/16/2018 - 09/17/2018

# ------------------------------------------------------------------------------------


import numpy as np
from scipy.special import beta
from math import log

# ---------------------------------------------------------------------------------------
#
#  function definition and invocation
#
# ---------------------------------------------------------------------------------------

def book_keeping_s2_m0(s2):

    n2 = np.array([])

    for i in s2:

        if i == 1:
            n2 = np.append(n2,1)
        else:
            n2[-1] += 1

    return n2


def book_keeping_s2_m1(s1, s2):

    n2 = np.array([])

    if len(s1) == len(s2):

        for i in range(0, len(s1)):

            if s2[i] == 1:
                n2 = np.append(n2, 1)
            elif s1[i] == s2[i]:                # s1[i] = s2[i] = 0
                n2[-1] += 1
            else:                               # s1[i] = 1 and s2[i] = 0
                n2[-1] = -n2[-1]                # minus sign marks that it should be beta function
                n2 = np.append(n2, 1)

    else:

        print("error!! len(s1) != len(s2)!! --function: book_keeping_s2_m1")
        return -1

    return n2


def YS_seq_likelihood(n, alpha):

    p = 1

    for i in n:

        if i > 0:

            p *= alpha * beta(i, alpha+1)

        else:

            p *= alpha * beta(-i, alpha)

    return p


def main():

    alpha_1 = 0.75
    alpha_2 = 0.75
    T = 2000              # length of time series


    # ---------------------------------------------------------------------------------------
    #
    #  generate random signals according to Yule-Simon process
    #
    # ---------------------------------------------------------------------------------------

    # in this example A is influencing B, only
    # no white noise signal

    # variables

    s1 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
    s2 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing

    s1[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1
    s2[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1

    n_count_1 = 1         # number of nodes in the current regime, agent 1
    n_count_2 = 1         # number of nodes in the current regime, agent 2



    # M1
    for t in range(1,T):

        # update value of p (probability of creating new regime)
        p1 = alpha_1 / (n_count_1 + alpha_1)
        p2 = alpha_2 / (n_count_2 + alpha_2)

        # if s[t] = 1, then x[t] belongs to a new regime
        s1[t] = np.random.binomial(1, p1)
        s2[t] = np.random.binomial(1, p2)

        # repeat:
        # n_count: update the number of nodes in the current regime

        if s1[t] == 1:
            n_count_1 = 1
            n_count_2 = 1

        else:
            n_count_1 = n_count_1 + 1

            if s2[t] == 1:
                n_count_2 = 1
            else:
                n_count_2 = n_count_2 + 1


    # #M0
    # for t in range(1,T):
    #
    #     # update value of p (probability of creating new regime)
    #     p1 = alpha_1 / (n_count_1 + alpha_1)
    #     p2 = alpha_2 / (n_count_2 + alpha_2)
    #
    #     # if s[t] = 1, then x[t] belongs to a new regime
    #     s1[t] = np.random.binomial(1, p1)
    #     s2[t] = np.random.binomial(1, p2)
    #
    #     # repeat:
    #     # n_count: update the number of nodes in the current regime
    #
    #     if s1[t] == 1:
    #         n_count_1 = 1
    #
    #     else:
    #         n_count_1 = n_count_1 + 1
    #
    #
    #     if s2[t] == 1:
    #         n_count_2 = 1
    #     else:
    #         n_count_2 = n_count_2 + 1

    # ---------------------------------------------------------------------------------------
    #
    #  infer the signals using Gibbs sampling algorithm
    #
    # ---------------------------------------------------------------------------------------


    # ---------------------- #
    #  1. initialization     #
    # ---------------------- #

    # in this example, A is influencing B

    # initial estimation values of parameters
    alpha_e_m0 = 0.75
    alpha_e_m1 = 0.75
    a_e = 1
    b_e = 1
    c_e = 1
    d_e = 1

    s_e_1 = np.zeros(T)     # inference: indicator of new regime
    s_e_2 = np.zeros(T)

    n_count_e_1 = 1         # inference: number of nodes in the current regime
    n_count_e_2 = 1

    s_e_1[0] = 1
    s_e_2[0] = 1

    # ---------------------- #
    #  2.    inference       #
    # ---------------------- #

    rep_alpha = 1000    # rounds of Gibbs sampler for alpha


    # draw alpha Gibbs sampler

    # consider: influence from A to B

    # M0: no influence assumption

    n2m0 = book_keeping_s2_m0(s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j
        # draw alpha

        b_draw_alpha = b_e

        for i in range(0,len(n2m0)):

            w = np.random.beta(a=alpha_e_m0 + 1, b=n2m0[i], size=1)

            b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n2m0)

        alpha_e_m0 = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)



    # M1: p=1 influence assumption

    n2m1 = book_keeping_s2_m1(s1, s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j
        # draw alpha

        b_draw_alpha = b_e

        for i in range(0,len(n2m1)):

            if n2m1[i] > 0:

                w = np.random.beta(a=alpha_e_m1 + 1, b=n2m1[i], size=1)

                b_draw_alpha = b_draw_alpha - log(w)

            else:

                w = np.random.beta(a=alpha_e_m1, b=-n2m1[i], size=1)

                if w == 0:
                    w = 0.0000000000001


                b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n2m1)

        alpha_e_m1 = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)


    # ---------------------------------------------------------------------------------------
    #
    #  calculate likelihood for each model
    #
    # ---------------------------------------------------------------------------------------

    pm0 = YS_seq_likelihood(n2m0, alpha_e_m0)

    pm1 = YS_seq_likelihood(n2m1, alpha_e_m1)



    # ---------------------------------------------------------------------------------------
    #
    #  plot the data
    #
    # ---------------------------------------------------------------------------------------
    return(alpha_e_m0, alpha_e_m1, pm0, pm1)



if __name__ == "__main__":

    alpha_e_m0, alpha_e_m1, pm0, pm1 = main()
    print(alpha_e_m0, alpha_e_m1, pm0, pm1)




