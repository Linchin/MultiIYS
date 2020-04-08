# coding: utf-8
"""
Detect the i-YS process with two sources of influence.

Date: 05/31/2019
Author: Lingqing Gan @ Stony Brook University

Problem:
As in the following figure, node A is the object of interest.
We want to decide whether node B is influencing node A via i-YS;
Independent from this, we also want to decide whether node C is
influencing node A via i-YS.

   ╭───╮      ╭───╮
   | B |      | C |
   ╰─┬─╯      ╰─┬─╯
    i-YS?     i-YS?
     └───┐ ┌────┘
         ↓ ↓
        ╭┴─┴╮
        | A |
        ╰───╯

M0: node A is not i-YS influenced by node B or C
M1: node A is i-YS influenced by node B, but not node C
M2: node A is i-YS influenced by node C, but not node B
M3: node A is i-YS influenced by both node B and node C

Program Structure:
 * Signal generation
 * Model detection

Notes:
 * Start with deterministic model.


Ref:
 * legc10_YSP_inter_two_influence.py
 * legc02_YSP_inter_bi.py

"""

from math import log
import numpy as np
from scipy.special import beta


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
    """ Calculate the likelihood of the sequence.
    Input:
        n (list): the length of each regime for the pertaining YS process
                n[i] > 0: the regime follows YS process
                n[i] < 0: the regime follows i-YS process
        alpha (float): the estimated value of alpha via Gibbs sampling

    Output:
        p (float): the likelihood given the sequence
    """

    p = 1

    for i in n:

        if i > 0:

            p *= alpha * beta(i, alpha+1)

        else:

            p *= alpha * beta(-i, alpha)

    return p


def main():

    # 1 --> node A
    # 2 --> node B
    # 3 --> node C

    alpha_1 = 0.75
    alpha_2 = 0.75
    alpha_3 = 0.75

    T = 2000              # length of time series

    influencing_model = "M0"   # "M0", "M1", "M2", "M3"


    # ---------------------------------------------------------------------------------------
    #
    #  generate random signals according to Yule-Simon process
    #
    # ---------------------------------------------------------------------------------------

    # variables

    s1 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
    s2 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
    s3 = np.zeros(T)

    s1[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1
    s2[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1
    s3[0] = 1

    n_count_1 = 1         # number of nodes in the current regime, agent 1
    n_count_2 = 1         # number of nodes in the current regime, agent 2
    n_count_3 = 1

    if influencing_model == "M0":
        # M0
        # agent 2 =\=> agent 1
        # agent 3 =\=> agent 1

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

            else:
                n_count_1 = n_count_1 + 1


            if s2[t] == 1:
                n_count_2 = 1

            else:
                n_count_2 = n_count_2 + 1


            if s3[t] == 1:
                n_count_3 = 1

            else:
                n_count_3 = n_count_3 + 1


    elif influencing_model == "M1":

        # M1
        # agent 2 ===> agent 1
        # agent 3 =\=> agent 1
        for t in range(1, T):

            # update value of p (probability of creating new regime)
            p1 = alpha_1 / (n_count_1 + alpha_1)
            p2 = alpha_2 / (n_count_2 + alpha_2)
            p3 = alpha_3 / (n_count_3 + alpha_3)

            # if s[t] = 1, then x[t] belongs to a new regime
            s1[t] = np.random.binomial(1, p1)
            s2[t] = np.random.binomial(1, p2)
            s3[t] = np.random.binomial(1, p3)

            # repeat:
            # n_count: update the number of nodes in the current regime

            if s1[t] == 1:
                n_count_1 = 1
            else:
                n_count_1 = n_count_1 + 1

            if s2[t] == 1:
                n_count_2 = 1
                n_count_1 = 1
            else:
                n_count_2 = n_count_2 + 1

            if s3[t] == 1:
                n_count_3 = 1
            else:
                n_count_3 = n_count_3 + 1




    elif influencing_model == "M2":

        # M2
        # agent 2 =\=> agent 1
        # agent 3 ===> agent 1
        for t in range(1, T):

            # update value of p (probability of creating new regime)
            p1 = alpha_1 / (n_count_1 + alpha_1)
            p2 = alpha_2 / (n_count_2 + alpha_2)
            p3 = alpha_3 / (n_count_3 + alpha_3)

            # if s[t] = 1, then x[t] belongs to a new regime
            s1[t] = np.random.binomial(1, p1)
            s2[t] = np.random.binomial(1, p2)
            s3[t] = np.random.binomial(1, p3)

            # repeat:
            # n_count: update the number of nodes in the current regime

            if s3[t] == 1:
                n_count_1 = 1
                n_count_3 = 1

            else:
                n_count_3 = n_count_3 + 1

                if s1[t] == 1:
                    n_count_1 = 1
                else:
                    n_count_1 = n_count_1 + 1

            if s2[t] == 1:
                n_count_2 = 1
            else:
                n_count_2 = n_count_2 + 1

    elif influencing_model == "M3":

        # M2
        # agent 2 ===> agent 1
        # agent 3 ===> agent 1
        for t in range(1,T):

            # update value of p (probability of creating new regime)
            p1 = alpha_1 / (n_count_1 + alpha_1)
            p2 = alpha_2 / (n_count_2 + alpha_2)
            p3 = alpha_3 / (n_count_3 + alpha_3)

            # if s[t] = 1, then x[t] belongs to a new regime
            s1[t] = np.random.binomial(1, p1)
            s2[t] = np.random.binomial(1, p2)
            s3[t] = np.random.binomial(1, p3)

            # repeat:
            # n_count: update the number of nodes in the current regime

            if s2[t] == 1:
                n_count_2 = 1
                n_count_1 = 1


            elif s3[t] == 1:
                n_count_3 = 1
                n_count_1 = 1

            else:
                n_count_3 = n_count_3 + 1
                n_count_2 = n_count_2 + 1

                if s1[t] == 1:
                    n_count_1 = 1
                else:
                    n_count_1 = n_count_1 + 1

        else:
            print("Model name wrong!")
            exit(-1)


    # ---------------------------------------------------------------------------------------
    #
    #  infer the signals using Gibbs sampling algorithm
    #
    # ---------------------------------------------------------------------------------------

    # ---------------------- #
    #  1. initialization     #
    # ---------------------- #

    # initial estimation values of parameters

    alpha_e_m0 = 0.75
    alpha_e_m1 = 0.75
    a_e = 1
    b_e = 1

    s_e_1 = np.zeros(T)     # inference: indicator of new regime
    s_e_2 = np.zeros(T)
    s_e_1[0] = 1
    s_e_2[0] = 1

    # ---------------------- #
    #  2.    inference       #
    # ---------------------- #

    rep_alpha = 1000    # rounds of Gibbs sampler for alpha


    # Estimate alpha using Gibbs sampler

    # M0:

    n2m0 = book_keeping_s2_m0(s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j

        b_draw_alpha = b_e

        for i in range(0,len(n2m0)):

            w = np.random.beta(a=alpha_e_m0 + 1, b=n2m0[i], size=1)

            b_draw_alpha = b_draw_alpha - log(w)

        a_draw_alpha = a_e + len(n2m0)

        # draw alpha

        alpha_e_m0 = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)



    # M1:

    n2m1 = book_keeping_s2_m1(s1, s2)

    for rep_alpha_index in range(0,rep_alpha):

        # draw w_j


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

        # draw alpha

        a_draw_alpha = a_e + len(n2m1)

        alpha_e_m1 = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)


    # ---------------------------------------------------------------------------------------
    #
    #  calculate likelihood for each model
    #
    # ---------------------------------------------------------------------------------------

    pm0 = YS_seq_likelihood(n2m0, alpha_e_m0)

    pm1 = YS_seq_likelihood(n2m1, alpha_e_m1)


    return(alpha_e_m0, alpha_e_m1, pm0, pm1)


if __name__ == "__main__":

    alpha_e_m0, alpha_e_m1, pm0, pm1 = main()
    print(alpha_e_m0, alpha_e_m1, pm0, pm1)

















