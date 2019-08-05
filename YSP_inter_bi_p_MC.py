__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      YSP_inter_bi_p_MC.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.

#                   Adding: two variables where one is influencing another

#                   10/11/2018     incoporating p and tackling the technical problems

#                   Since the problem arises when there are multiple s_A=1 in between
#                   two s_B = 1, we use a Monte Carlo method to achieve an approximated
#                   solution. An adjustable number $MC_trace of traces are generated randomly
#                   where each trace shows wether A has affected B or not. With each trace,
#                   we use Gibbs sampling to acquire the value of alpha. We use imperfect
#                   Bernoulli trial to calculate the a posterior distribution of p, which
#                   turns out to be a polynomial function.
#                   Since we use the method of moments to approximate the polynomial posterior
#                   into the form of Beta distribution, the mean and variance of the resulting
#                   traces can be conveniently combined, in the way of a linear combination.
#                   Similarly, the estimated value of alpha is also taken as the mean of the
#                   resulting value of alpha of Gibbs sampling from each trace.



#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           10/11/2018 - 10/15/2018

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

def current_seq_massage(s2, amb_seq, current_index):

    n2_massage = np.array([])

    for i in range(0,len(s2)):

        if s2[i] == 1:
            n2_massage = np.append(n2_massage,1)

        elif i in amb_seq:

            flag = 0

            for j in range(0, len(amb_seq)):

                if amb_seq[j] == i:

                    if current_index[j] == 1:
                        flag = 1
                        break

                else:
                    continue

            if flag == 1:

                n2_massage[-1] = -n2_massage[-1]  # minus sign marks that it should be beta function
                n2_massage = np.append(n2_massage, 1)

        else:
            n2_massage[-1] += 1


    return n2_massage




def book_keeping_s2_mp(s1, s2):

    # randomly generate a trace which provides a hidden reset trace of s2

    n2 = np.array([])

    flag = 0


    if len(s1) == len(s2):

        for i in range(0, len(s1)):

            if s2[i] == 1 and flag == 0:

                n2 = np.append(n2, 1)

            elif s1[i] == 1 and s2[i] == 0 and flag == 0:

                n_temp = np.array([n2[-1],1])

                flag = 1

            elif s1[i] == 1 and s2[i] == 0 and flag == 1:

                n_temp = np.append([n2[-1], 1])

            elif s2[i] == 1 and flag == 1:

                flag = 0

                for item in n_temp:

                    n2 = np.append(n2,-item)

            elif s1[i] == 0 and flag == 1:

                n_temp[-1] += 1

            else:
                # s1[i] = s2[i] = 0
                n2[-1] += 1

    else:

        print("error!! len(s1) != len(s2)!! --function: book_keeping_s2_mp")
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



def identify_ambiguous_indices(s1,s2):

    index = np.array([])

    if len(s1) == len(s2):

        flag = 0

        for i in range(0,len(s1)):

            if s1[i] == 1 and s2[i] == 0:

                index = np.append(index,i)

        return index

    else:

        print("error!! len(s1) != len(s2)!! --function: book_keeping_s2_mp")
        return -1


def main():

    # hyper parameters

    a = 1
    b = 1
    c = 1
    d = 1

    T = 100              # length of time series


    # ---------------------------------------------------------------------------------------
    #
    #  generate random signals according to Yule-Simon process
    #
    # ---------------------------------------------------------------------------------------

    # in this example A is influencing B, only
    # no white noise signal

    p = 0.5


    alpha_1 = 0.75
    alpha_2 = 0.75

    # variables

    s1 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing
    s2 = np.zeros(T)     # indicator of new regime. s[t] = 1 means new regime, s[t] = 0 means continue existing

    s1[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1
    s2[0] = 1            # the first node is automatically assigned to the first regime, so new regime indicatior is 1

    n_count_1 = 1         # number of nodes in the current regime, agent 1
    n_count_2 = 1         # number of nodes in the current regime, agent 2



    # Mp (always M1 now, since we use the value of p)
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

            dice = np.random.binomial(1, p)
            if dice == 1:
                n_count_2 = 1

        else:
            n_count_1 = n_count_1 + 1

            if s2[t] == 1:
                n_count_2 = 1
            else:
                n_count_2 = n_count_2 + 1


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

    a_e = 1
    b_e = 1

    s_e_1 = np.zeros(T)     # inference: indicator of new regime
    s_e_2 = np.zeros(T)

    s_e_1[0] = 1
    s_e_2[0] = 1

    p_alpha = 1
    p_beta = 1

    alpha_e_m1 = 0.75

    rep_alpha = 400  # rounds of Gibbs sampler for alpha

    # identify ambiguous situations

    amb_seq = identify_ambiguous_indices(s1,s2)             # list of ambiguous sequences
    print(len(amb_seq))

    # save the value of alpha, prob of each trace, and the polynomial values

    alpha_per_sequence = np.zeros([2**len(amb_seq)])
    prob_per_sequence = np.zeros([2**len(amb_seq)])
    beta_dist_alpha_per_sequence = np.zeros([2**len(amb_seq)])
    beta_dist_beta_per_sequence = np.zeros([2 ** len(amb_seq)])

    # ---------------------- #
    #  2.    inference       #
    # ---------------------- #

    for MC_rep_index in range(0,2**len(amb_seq)):

        print("round:" + str(MC_rep_index))

        # 1. generate the current sequence and bookkeeping it

        current_index = bin(MC_rep_index)[2:].zfill(len(amb_seq)) # binary sequence

        current_index_list = [int(item) for item in str(current_index)]

        n2_current_seq = current_seq_massage(s2, amb_seq, current_index_list)


        # 2. Gibbs sampling for alpha

        for rep_alpha_index in range(0,rep_alpha):

            # draw w_j
            # draw alpha

            b_draw_alpha = b_e

            for i in range(0,len(n2_current_seq)):

                if n2_current_seq[i] > 0:

                    w = np.random.beta(a=alpha_e_m1 + 1, b=n2_current_seq[i], size=1)

                    b_draw_alpha = b_draw_alpha - log(w)

                else:

                    w = np.random.beta(a=alpha_e_m1, b=-n2_current_seq[i], size=1)

                    if w == 0:
                        w = 0.0000000000001


                    b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n2_current_seq)

            alpha_e_m1 = np.random.gamma(shape=a_draw_alpha, scale = 1 / b_draw_alpha)

        alpha_per_sequence[MC_rep_index] = alpha_e_m1


        # 3. posterior distribution of p

        beta_dist_alpha_per_sequence[MC_rep_index] = p_alpha + sum(current_index_list)
        beta_dist_beta_per_sequence[MC_rep_index] = p_beta + len(amb_seq) - sum(current_index_list)


        # 4. calculate likelihood for each model

        pm_p = YS_seq_likelihood(n2_current_seq, alpha_e_m1)

        prob_per_sequence[MC_rep_index] = pm_p


    print("Start to save data...")
    # save all the data

    with open('YSP_inter_bi_p_MC_result_10152018_prob.txt', 'w') as f:
        for item in prob_per_sequence:
            f.write("{} \n".format(item))

    with open('YSP_inter_bi_p_MC_result_10152018_alpha.txt', 'w') as f:
        for item in alpha_per_sequence:
            f.write("{} \n".format(item))

    with open('YSP_inter_bi_p_MC_result_10152018_beta_dist_alpha.txt', 'w') as f:
        for item in beta_dist_alpha_per_sequence:
            f.write("{} \n".format(item))

    with open('YSP_inter_bi_p_MC_result_10152018_beta_dist_beta.txt', 'w') as f:
        for item in beta_dist_beta_per_sequence:
            f.write("{} \n".format(item))

    print("Finished saving data. Have a nice day! :-)")


if __name__ == "__main__":

    main()





