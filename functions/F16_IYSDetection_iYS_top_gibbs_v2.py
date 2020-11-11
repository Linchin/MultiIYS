"""
A class that detects the i-YS relationship.

File name:
F16_IYSDetection_iYS_top_gibbs_v2.py

Class name:
IYSDetection_iYS_top_gibbs

Author:
Lingqing Gan @ Stony Brook University

10/07/2020 notes (F16)
* add attribute that returns the estimated adjacency matrix
  at every Gibbs sampling iteration.
* add attribute that returns the estimated rho value
  at every Gibbs sampling iteration
  (just use the last rho value )


04/11/2020 notes (F14)
This code is going to implement a new idea.

For each node i
    for each neighbor j != i:
        do gibbs sampling suppose other remain the same
Gibbs sampling for alpha



01/06/2020 notes (F13)
This version corrected the function for iYS prob.

12/29/2019 notes (F12)
This F12 version is based on F11, which is a multi-node stable
IYS Detection class. This version (F12) is created in correspondence
with exp18, where we change the condition of stopping model selection
from reaching a given number of total time slots into reaching a given
number of regimes for each node-neighbor pair. This helps with the problem
that different nodes may have different regime density, so with the same
number of time slots, we have drastically varying number of regimes with
each node. With a too small number of regimes we cannot draw a convincing
model selection result. With too much regimes we generate a likelyhood so
small that the program can no longer handle (though we can solve that in
some other way - to normalize the likelihoods at each regime, and we may
do that some time in the future).

12/02/2019 notes (F11)
The F10 version is working now. F11 is a copy of it which we will
make sure it works on multi-node scenarios.

11/07/2019 update
I realized that the algo used to parse the data is not correct. Now
attempt to change it.

10/31/2019 note
this node is in accordance with the stable_02 series,
where the first instant all signals are 1 instead of 0.

10/22/2019 update
* Finished writing the separation and corresponding update process;
* Now start debugging.

10/10/2019 update
* Now we call the separated signals "unambiguous". :)
* Sorted out the data structure.
* Wrote down the data struct in the notebook.
* Wrote down the program outline in the notebook.

09/19/2019 update
F10_IYSDetection_parse.py

Update 09/23/2019:
Plan of changes to this version:
1. parse all the sequences so that only "pure" sequences are considered.
2. a pure sequence: the regimes of a node during which only one of the
    neighbors have changed their regime. Thus during this regime, the node
    of interest can only be influenced by one neighbor.
3. If there is no new regime among the neighbors during a regime of the
    node of interest, we also keep track of it. This helps us better
    estimate the value of rho for the node of interest.
4. We would use the detection class to keep track of the number of pure
    sequences for each model for each node of interest. This would be
    important to understanding the required length of time instants
    in order to acquire sufficient data to carry out the Gibbs sampling
    procedure.
5. A parameter: effective_regime_number
    designated number of regimes that need to be satisfied to make the
    Gibbs sampling procedure effective. e.g. effective_regime_number = 20

We parse the signal sequences so that with one influenced node,
we only consider the intervals where this node is only possible to be
affected by ONE single neighbor.

08/01/2019
added the history of estimated rho value.
"""

from math import log
from typing import List, Dict
from sys import stdout

import numpy as np
import scipy.special


class IYSDetection_iYS_top_gibbs:
    """
    Detect the i-YS relationship on the network.
    We parse the sequence so we only use the unambiguous sequence for
    detection.
    """
    def __init__(self, network_size, gibbs_rep, gibbs_alpha_rep, required_time):
        """
        Create an i-YS relationship detection object.
        """
        # --------------------------------------------------------------
        # 1. Parameters set up
        # --------------------------------------------------------------

        # ADDED IN F14:
        self.__gibbs_rep = gibbs_rep        # rounds of Gibbs samplers for topology

        self.__rep_alpha = gibbs_alpha_rep  # rounds of Gibbs sampler for alpha
        self.__network_size = network_size  # number of nodes in the network

        # REMOVED IN F14:
        # indicator of starting new regime at the new time instant
        # 0: no; 1: new
        # self.__new_regime_indicator = np.zeros((self.__network_size, 1))

        # time instant starting from 0
        self.__network_time = -1

        # ADDED IN F14:
        # REMOVED IN F12:
        # total time instants to be observed
        # (to save the time of calculation, we only start the detection
        # process when the total time instants is reached.)
        self.__total_time_instant = required_time

        # REMOVED IN F14:
        # ADDED IN F12:
        # for each node-neighbor pair, after reaching the given number of
        # regimes, we make the model selection and stop the model selection
        # for this pair.
        # self.__required_regimes = required_regimes

        # REMOVED IN F14:
        # flag: if True, all node-neighbor pairs have reached the given
        # threshold self.__required_regime
        # readable through @property
        # self.__all_regime_reached = False

        # REMOVED IN F14:
        # matrix that stores the status of each node-neighbor pair
        # 0: not yet reached the self.__required_regime
        # 1: reached the self.__required_regime
        # self.__regime_status = np.zeros((self.__network_size,
        #                                 self.__network_size))

        # ----------------------------------------------------------------
        # 2. Construct data structures to store all necessary history data
        # ----------------------------------------------------------------
        # All used dictionaries:
        # Dict: self.__signal_history
        # signal history of the entire network
        # Dict: self.__likelihood_history[i][j]
        # store the likelihood history of each model as a list
        # Dict: self.__aprob_history[i][j]
        # store the a posterior prob history of each model
        # Dict: self.__rho_history[i][j]
        # store the a posterior prob history of each model
        # Dict: self.__pure_regime[i][j]
        # store the number and length of pure regimes for each node&model
        # pure_regime[i][i] registers the number of pure regimes
        # of node i during which no other nodes started a new regime
        # Dict: __regime_shift _time[i] - list
        # stores the index of regime shift of node i
        # Dict: __unambi_regime_count
        # stores the number and length of unambiguous regimes
        # List: __ambi_regime_count
        # stores the number of ambiguous regimes for each node
        # i: index of the node of interest
        # j: the node that is likely to influence node i
        # Dict: __combined_signals
        # stores the combined unambiguous signal that we get after parsing.
        # i: the node of interest
        # j: the node likely to be the influencer.
        # Dict: __bookkeeping_results
        # stores the signal bookkeeping results
        # i: the node of interest
        # j: the node likely to be the influencer.
        # Dict: __adjacency_mat_e_history
        # i: the Gibbs iteration

        self.__signal_history = {}      # type: Dict[int: List[Bool]]
        # self.__likelihood_history = {}  # type: Dict[int: Dict[int: List[int]]]
        # self.__aprob_history = {}       # type: Dict[int: Dict[int: List[int]]]
        self.__rho_history = []         # type: List
        # self.__pure_regime = {}         # type: Dict[int: Dict[int: List[tuple]]]
        # self.__regime_shift_time = {}   # type: Dict[int: List[int]]
        # self.__unambi_regime_count = {}  # type: Dict[int: List[int]]
        # self.__ambi_regime_count = 0    # type: int
        # self.__combined_signals = {}    # type: Dict[int: Dict[int: Dict["m0"/"m1": List]]]
        # self.__bookkeeping_results = {}  # type: Dict[int: Dict[int: Dict["m0"/"m1": array]]]
        self.__adjacency_mat_e_history = []  # type: List

        for i in range(0, self.__network_size):
            self.__signal_history[i] = []
            # self.__likelihood_history[i] = {}
            # self.__aprob_history[i] = {}
            # self.__rho_history[i] = []
            # self.__pure_regime[i] = {}
            # self.__regime_shift_time[i] = []
            # self.__unambi_regime_count[i] = []
            # self.__combined_signals[i] = {}
            # self.__bookkeeping_results[i] = {}
            # for j in range(0, self.__network_size):
                # self.__likelihood_history[i][j] = []
                # self.__aprob_history[i][j] = []
                # self.__pure_regime[i][j] = []    # tuple (T, t)
                # self.__unambi_regime_count[i].append(0)
                # self.__combined_signals[i][j] = {}  # type: Dict["m0"/"m1": List]
                # self.__bookkeeping_results[i][j] = {}  # type: Dict["m0"/"m1": List]

    # ----------------------------------------------------------------
    # API: methods for read only parameters
    # ----------------------------------------------------------------
    # @property
    # def likelihood_history(self):
    #     return self.__likelihood_history

    # @property
    # def aprob_history(self):
    #     return self.__aprob_history

    @property
    def rho_history(self):
        return self.__rho_history

    # @property
    # def pure_regime(self):
    #     return self.__pure_regime

    # @property
    # def regime_shift_time(self):
    #     return self.__regime_shift_time

    @property
    def signal_history(self):
        return self.__signal_history

    # @ property
    # def combined_signals(self):
    #     return self.__combined_signals

    # @property
    # def regime_reached(self):
    #     return self.__all_regime_reached

    @property
    def adjacency_mat_e_history(self):
        return self.__adjacency_mat_e_history

    def read_new_time(self, new_col):
        """
        *Callable*
        method that
        1) reads the new signal of the network,
        2) parse the data,
        3) save the data to the structs in this object.
        4) call self.__estimate_update() to update the model likelihood.
        """

        # ----------------------------------------------------
        #  1st time instant
        # ----------------------------------------------------

        if self.__network_time == -1:

            self.__network_time = 0
            # self.__new_regime_indicator = np.ones(
            #    (self.__network_size, 1))

            for i in range(0, self.__network_size):
                self.__signal_history[i].append(1)
            #    self.__regime_shift_time[i].append(0)

                # 01/02/2020 change:
                # remove the 1st mandatory time slot from
                # counting as an unambiguous regime

                # self.__unambi_regime_count[i][i] += 1
                # signals_temp_self = np.ones(2)
                # self.__pure_regime[i][i].append(signals_temp_self)

            # F12: no longer need to estimate at the first time slot
            # self.__estimate_update()
            return 0

        # ----------------------------------------------------
        #  2nd time instant and later
        # ----------------------------------------------------

        self.__network_time += 1
        # self.__new_regime_indicator = np.copy(new_col)

        # update the regime history
        for i in range(0, self.__network_size):

            # register the latest signals
            self.__signal_history[i].append(new_col[i])

            # skip if the current node doesn't have a new regime
            # if new_col[i] == 0:
            #    continue

            # register the latest regime shift time
            # self.__regime_shift_time[i].append(self.__network_time)

            # # decide if this new regime is ambiguous or unambiguous
            # # (0/1/2+ influencers)
            # begin = self.__regime_shift_time[i][-2]
            # end = self.__regime_shift_time[i][-1]
            # count_parse = 0   # counter of number of influencing neighbors
            # for j in range(0, self.__network_size):
            #     if j == i:
            #         continue
            #     last_rgm_shft = self.__regime_shift_time[j][-1]
            #     if last_rgm_shft == end:
            #         last_rgm_shft = self.__regime_shift_time[j][-2]
            #     if begin < last_rgm_shft <= end-1:
            #         count_parse += 1
            #         influencer = j
            #
            # # case 1: there is 0 influencing neighbor
            # if count_parse == 0:
            #     # if there are enough unambiguous regimes for this node,
            #     # just skip it
            #     if self.__regime_status[i][i] == 1:
            #         continue
            #     self.__unambi_regime_count[i][i] += 1
            #     # reconstruct the self influencing signals
            #     signals_temp_self = np.zeros(end-begin+1)
            #     signals_temp_self[0] = 1
            #     signals_temp_self[-1] = 1
            #     self.__pure_regime[i][i].append(signals_temp_self)
            #
            #     # check if this node has enough unambiguous regimes
            #     if self.__unambi_regime_count[i][i] == self.__required_regimes:
            #         self.__regime_status[i][i] = 1
            #         # check if all node-neighbor pairs have enough unambiguous regimes
            #         if np.sum(self.__regime_status) == self.__network_size**2:
            #             self.__all_regime_reached = True
            #             break
            #
            # # case 2: there is exactly 1 possible influencer
            # elif count_parse == 1:
            #     # if this pair has enough regimes, just skip it
            #     if self.__regime_status[i][influencer] == 1:
            #         continue
            #     self.__unambi_regime_count[i][influencer] += 1
            #     # reconstruct the signals
            #     regime_recon = np.zeros(end-begin+1)      # influenced node
            #     regime_recon[0] = 1
            #     regime_recon[-1] = 1
            #     relative_inf_time = []  # include all relative influencing time
            #     # find out the list of relative influencing time
            #     for item in self.__regime_shift_time[influencer]:
            #         if begin < item <= end-1:
            #             relative_inf_time.append(item-begin)
            #     self.__pure_regime[i][influencer].append((relative_inf_time, regime_recon))
            #     # (length of the current regime; relative time point of
            #     # the possible influence)
            #
            #     # check if this node-neighbor pair has enough unambiguous regimes
            #     if self.__unambi_regime_count[i][influencer] == self.__required_regimes:
            #         self.__regime_status[i][influencer] = 1
            #         # check if all node-neighbor pairs have enough unambiguous regimes
            #         if np.sum(self.__regime_status) == self.__network_size**2:
            #             self.__all_regime_reached = True
            #             break
            #
            # # case 3: there are at least 2 possible influencers
            # else:
            #     self.__ambi_regime_count += 1

        # print the number of unambiguous regimes for each node
        # and possible influencing nodes
        # printed as a matrix
        # stdout.write("Total time instants: %d; \nTotal ambiguous regimes: %d; \n"
        #             % (self.__network_time,
        #                self.__ambi_regime_count))
        # print_string = "Unambiguous regimes # of node %d: " + \
        #               "%d, " * (self.__network_size-1) + "%d" + "\n"
        # for i_temp in range(0, self.__network_size):
        #    print_data = tuple([i_temp]) + tuple(self.__unambi_regime_count[i_temp][j]
        #                       for j in range(0,self.__network_size))
        #    stdout.write(print_string % print_data)

        # if the entire network has enough unambiguous regimes,
        # proceed with iYS model selection
        # if self.__all_regime_reached is True:
        if self.__network_time == self.__total_time_instant-1:
            adj_mat_e = self.__estimate_update()
            print(adj_mat_e)

        return 0

    def __estimate_update(self):
        """
        Function that carries out the estimation algorithm.

        If there is a new regime for any of the nodes,
        we carry out the estimation algo.
        This is a version that saves time:
        we only start the estimation (Gibbs sampling) when the
        last time instant is reached.

        :return: the a posterior prob list for each node and their neighbor
        """

        current_adj_matrix = np.zeros((self.__network_size,
                                       self.__network_size))

        alpha_e = 1

        for gibbs_rep_index in range(self.__gibbs_rep):
            # Gibbs sampling for topology

            print(current_adj_matrix)
            print("Gibbs rep: ", gibbs_rep_index)

            # save the result of the last gibbs iteration
            self.__adjacency_mat_e_history.append(current_adj_matrix.copy())
            self.__rho_history.append(alpha_e)

            for i in range(0, self.__network_size):
                for j in range(0, self.__network_size):
                    # print(i, j)

                    if i == j:
                        continue

                    # 1) Generate the sequence related to node i & j
                    #    given influenced/not influenced

                    # adj matrix given H0
                    adj_matrix_0 = np.copy(current_adj_matrix)
                    adj_matrix_0[i, j] = 0

                    # adj matrix given H1
                    adj_matrix_1 = np.copy(current_adj_matrix)
                    adj_matrix_1[i, j] = 1

                    # generate sequence
                    n_m0 = self.__gibbs_sequence(i, adj_matrix_0)
                    n_m1 = self.__gibbs_sequence(i, adj_matrix_1)

                    # Calculate the likelihood of both models
                    lklhd_m0 = self.__ys_seq_likelihood(n_m0, alpha_e)
                    lklhd_m1 = self.__ys_seq_likelihood(n_m1, alpha_e)
                    # print(" Node of interest:", i, " Influencing node:", j,
                    #       " P(M_0):", lklhd_m0, " P(M_1):", lklhd_m1)

                    # Normalization
                    temp_norm = lklhd_m0 + lklhd_m1
                    aprob_m0, aprob_m1 = lklhd_m0 / temp_norm, lklhd_m1 / temp_norm

                    # Bernoulli random sampling
                    gibbs_random_sample = np.random.binomial(1, aprob_m1)
                    if gibbs_random_sample == 1:
                        current_adj_matrix[i, j] = 1
                    elif gibbs_random_sample == 0:
                        current_adj_matrix[i, j] = 0
                    else:
                        print("Wrong Bernoulli result. Exit.")
                        exit(-1)

                    # 5) Save the data
                    # self.__likelihood_history[i][j].append((lklhd_m0, lklhd_m1))
                    # self.__aprob_history[i][j].append((aprob_m0, aprob_m1))
                    # self.__rho_history[i][j].append(alpha_e)
                    # self.__combined_signals[i][j]["m0"] = s_combined_m0
                    # self.__combined_signals[i][j]["m1"] = s_combined_m1
                    # self.__bookkeeping_results[i][j]["m0"] = n_m0
                    # self.__bookkeeping_results[i][j]["m1"] = n_m1

                # Gibbs sampling for the alpha value of the ith node
                n_i = self.__gibbs_sequence(i, current_adj_matrix)
                alpha_e = self.__gibbs_sampling_alpha(n_i)



        # return the last adjacency matrix
        return current_adj_matrix


    def __gibbs_sequence(self, i, adj_matrix):
        # return the sequence for ith node given adj_matrix

        # extract the influencing pattern
        influencing_pattern = adj_matrix[int(i)]

        total_time = len(self.__signal_history[0])
        network_size = self.__network_size

        combined_signal = np.zeros(total_time)
        for time in range(total_time):

            if self.__signal_history[i][time] == 1:
                combined_signal[time] = 1
                continue

            for index in range(network_size):

                if influencing_pattern[index] == 0:
                    continue
                elif self.__signal_history[index][time]:
                    combined_signal[time] = -1
                    break

        seq = self.__book_keeping_from_time(combined_signal)

        return seq

    @staticmethod
    def __book_keeping_from_time(s_combined):
        """
        Return the bookkeeping sequence given the hypothesis that the neighbor
        has influence.
        s_sf: the list of regimes where the node is not influenced by anyone
              (self)
        s_nb: the list of regimes where the node is possible to be influenced
              by one neighbor.
              (neighbor)
              -- now it's the exact 0/1 signals after parsing but
              still need reconstruction.
        :return: n
        10/31/2019
        a significant alteration due to the parsing algo.
        the input is reconstructed signals instead of the numbers.
        This function does:
        1. Combine the reconstructed signals into a consecutive signal
            sequence.
        2. Convert the combined signals into the book keeping sequence.
        s_2: the reconstructed signal sequence of the influenced node
        s_1: the reconstructed signal sequence of the influencing node
        """

        # generate the sequence after book keeping
        # 01/02/2020
        # changed from [] to [1]
        if s_combined[0] == 0:
            n = np.array([0])
        else:
            n = np.array([])

        for i in range(0, len(s_combined)):
            if s_combined[i] == 1:
                n = np.append(n, 1)
            elif s_combined[i] == -1:
                n[-1] = -n[-1]
                n = np.append(n, 1)
            else:
                n[-1] += 1
        return n

    def __gibbs_sampling_alpha(self, n):
        # parameters
        b_e = 1
        a_e = 1
        alpha_e = 0.75

        for rep_alpha_index in range(0, self.__rep_alpha):
            # draw w_j
            # draw alpha
            b_draw_alpha = b_e

            for i in range(0, len(n)):

                if n[i] > 0:
                    # YS case
                    w = np.random.beta(a=alpha_e+1, b=n[i], size=1)
                    b_draw_alpha = b_draw_alpha - log(w)
                else:
                    # i-YS case
                    # 01/06/2020 change
                    # b from b=-n[i] into b=-n[i]+1
                    w = np.random.beta(a=alpha_e, b=-n[i]+1, size=1)
                    if w == 0:
                        w = 0.0000000000001
                    b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n)

            alpha_e = np.random.gamma(shape=a_draw_alpha, scale=1/b_draw_alpha)

        return alpha_e

    @staticmethod
    def __ys_seq_likelihood(n, alpha):
        """
        Returns the likelihood of a specific sequence.

        Problem: 20190820
        When the sequence is very long, the returned value of likelihood
        gets very small. And then it gets to zero.

        attempted solution:
        for each time instant, we multiply the likelihood by a constant > 1.

        Args:
            n: list
                The sequence.
            alpha: float in (0, 1)
                Estimated value of parameter alpha.
        Returns:
            p: float in (0, 1)
                The likelihood of sequence *n*.
        """
        p = 1.
        for i_l in n:
            if i_l > 0:
                p *= alpha * scipy.special.beta(i_l, alpha + 1)
            else:
                # 01/06/2020 change
                # changed the first parameter from -i_l into -i_l + 1
                p *= alpha * scipy.special.beta(-i_l + 1, alpha)
        return p


