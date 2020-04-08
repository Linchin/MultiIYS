__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      legc03_YSP_inter_bi_bidir.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.

#                   Adding: two variables where one is influencing another

#                   now A and B are influencing each other at the same time

#                   we want to see if our method could tell or not

#                   a modification of legc02_YSP_inter_bi.py


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/26/2018 - 09/26/2018

# ------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t as student
from scipy.special import beta
from math import sqrt
from math import log

# import sys
# print(sys.path)

# ---------------------------------------------------------------------------------------
#
#  function definition and invocation
#
# ---------------------------------------------------------------------------------------


def book_keeping_s2_m0(s2):
    """
    Return the sequence of regime length given M0.
    :param s2:
    :return nparray:
    """

    n2 = np.array([])

    for i in s2:

        if i == 1:
            n2 = np.append(n2,1)
        else:
            n2[-1] += 1

    return n2


def book_keeping_s2_m1(s1, s2):
    """
    Return the sequence of regime given M1.
    If the value in the returned list is negative, it means this regime ended because of a neighbor.
    :param s1:
    :param s2:
    :return nparray:
    """

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

    # hyper parameters

    a = 1
    b = 1
    c = 1
    d = 1

    T = 1000              # length of time series


    # ---------------------------------------------------------------------------------------
    #
    #  generate random signals according to Yule-Simon process
    #
    # ---------------------------------------------------------------------------------------

    # in this example A is influencing B, only
    # no white noise signal




    alpha_1 = 0.75
    alpha_2 = 0.75

    # variables

    s1 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
    s2 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing

    s1[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1
    s2[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1

    n_count_1 = 1         # number of nodes in the current regime, agent 1
    n_count_2 = 1         # number of nodes in the current regime, agent 2


    # M2            where A -> B and B -> A
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

        elif s2[t] == 1:
            n_count_1 = 1
            n_count_2 = 1

        else:
            n_count_1 = n_count_1 + 1
            n_count_2 = n_count_2 + 1

    #
    # # M1
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
    #         n_count_2 = 1
    #
    #     else:
    #         n_count_1 = n_count_1 + 1
    #
    #         if s2[t] == 1:
    #             n_count_2 = 1
    #         else:
    #             n_count_2 = n_count_2 + 1


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
    alpha_e_m0_AB = 0.75
    alpha_e_m1_AB = 0.75
    alpha_e_m0_BA = 0.75
    alpha_e_m1_BA = 0.75
    a_e = 1
    b_e = 1

    s_e_1 = np.zeros(T)     # inference: indicator of new regime
    s_e_2 = np.zeros(T)

    s_e_1[0] = 1
    s_e_2[0] = 1

    # ---------------------- #
    #  2.    inference       #
    # ---------------------- #

    rep_alpha = 100    # rounds of Gibbs sampler for alpha


    # draw alpha Gibbs sampler

    # ********************************

    # 1. influence from A to B

    # ********************************

    # 1.1 A->B   M0: no influence assumption

    n2m0_AB = book_keeping_s2_m0(s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j
        # draw alpha

        b_draw_alpha = b_e

        for i in range(0,len(n2m0_AB)):

            w = np.random.beta(a=alpha_e_m0_AB + 1, b=n2m0_AB[i], size=1)

            b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n2m0_AB)

        alpha_e_m0_AB = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)

    # 1.2 A->B   M1: p=1 influence assumption

    n2m1_AB = book_keeping_s2_m1(s1, s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j
        # draw alpha

        b_draw_alpha = b_e

        for i in range(0,len(n2m1_AB)):

            if n2m1_AB[i] > 0:

                w = np.random.beta(a=alpha_e_m1_AB + 1, b=n2m1_AB[i], size=1)

                b_draw_alpha = b_draw_alpha - log(w)

            else:

                w = np.random.beta(a=alpha_e_m1_AB, b=-n2m1_AB[i], size=1)

                if w == 0:
                    w = 0.0000000000001

                b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n2m1_AB)

        alpha_e_m1_AB = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)

        # ********************************

        # 2. influence from B to A

        # ********************************

        # 1.1 B->A   M0: no influence assumption

        n2m0_BA = book_keeping_s2_m0(s1)

        for rep_alpha_index in range(0, rep_alpha):

            # draw w_j
            # draw alpha

            b_draw_alpha = b_e

            for i in range(0, len(n2m0_BA)):
                w = np.random.beta(a=alpha_e_m0_BA + 1, b=n2m0_BA[i], size=1)

                b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n2m0_BA)

            alpha_e_m0_BA = np.random.gamma(shape=a_draw_alpha, scale=1 / b_draw_alpha)

        # 1.2 B->A   M1: p=1 influence assumption

        n2m1_BA = book_keeping_s2_m1(s2, s1)

        for rep_alpha_index in range(0, rep_alpha):

            # draw w_j
            # draw alpha

            b_draw_alpha = b_e

            for i in range(0, len(n2m1_BA)):

                if n2m1_BA[i] > 0:

                    w = np.random.beta(a=alpha_e_m1_BA + 1, b=n2m1_BA[i], size=1)

                    b_draw_alpha = b_draw_alpha - log(w)

                else:

                    w = np.random.beta(a=alpha_e_m1_BA, b=-n2m1_BA[i], size=1)

                    if w == 0:
                        w = 0.0000000000001

                    b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n2m1_BA)

            alpha_e_m1_BA = np.random.gamma(shape=a_draw_alpha, scale=1 / b_draw_alpha)

    # ---------------------------------------------------------------------------------------
    #
    #  calculate likelihood for each model
    #
    # ---------------------------------------------------------------------------------------

    # A->B

    pm0_AB = YS_seq_likelihood(n2m0_AB, alpha_e_m0_AB)

    pm1_AB = YS_seq_likelihood(n2m1_AB, alpha_e_m1_AB)

    # B->A

    pm0_BA = YS_seq_likelihood(n2m0_BA, alpha_e_m0_BA)

    pm1_BA = YS_seq_likelihood(n2m1_BA, alpha_e_m1_BA)

    return pm0_AB, pm1_AB, pm0_BA, pm1_BA


if __name__ == "__main__":

    print(main())





