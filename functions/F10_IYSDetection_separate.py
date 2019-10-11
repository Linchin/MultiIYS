"""
A class that detects the i-YS relationship.

09/19/2019 update
F10_IYSDetection_separate.py

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

Check:
1. working
2. working
3.
4.


We parse the signal sequences so that with one influenced node,
we only consider the intervals where this node is only possible to be
affected by ONE single neighbor.

08/01/2019
added the history of estimated rho value.
"""

from math import log
from typing import List, Dict

import numpy as np
import scipy.special


class IYSDetection_parse:
    """
    Detect the i-YS relationship on the network.
    """

    def __init__(self, network_size, gibbs_rep, time_inst):
        """
        Create an i-YS relationship detection object.
        """

        # --------------------------------------------------------------
        # 1. Parameters set up
        self.__rep_alpha = gibbs_rep  # rounds of Gibbs sampler for alpha

        # instance variables
        self.__network_size = network_size

        # signal history of the entire network
        self.__signal_history = {}
        for i in range(0, self.__network_size):
            self.__signal_history[i] = []

        # indicator of starting new regime at the new time instant
        # 0: no; 1: new
        self.__new_regime_indicator = np.zeros((self.__network_size, 1))

        # time instant starting from 0
        self.__network_time = -1

        # total time instants to be observed
        # (to save the time of calculation, we only start the detection
        # process when the total time instants is reached.)
        self.__total_time_instant = time_inst
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # 2. Construct data structures to store all necessary history data
        #
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

        self.__likelihood_history = {}  # type: Dict[int: Dict[int: List]]
        self.__aprob_history = {}       # type: Dict[int: Dict[int: List]]
        self.__rho_history = {}         # type: Dict[int: Dict[int: List]]
        self.__pure_regime = {}         # type: Dict[int: Dict[int: List]]
        self.__regime_shift_time = {}   # type: Dict[int: List]
        self.__unambi_regime_count = {} # type: Dict[[i][j]: int]
        self.__ambi_regime_count = []   # type: List

        for i in range(0, self.__network_size):
            self.__likelihood_history[i] = {}
            self.__aprob_history[i] = {}
            self.__rho_history[i] = {}
            self.__pure_regime[i] = {}
            self.__regime_shift_time[i] = []
            self.__unambi_regime_count[i] = []

            for j in range(0, self.__network_size):
                self.__likelihood_history[i][j] = []
                self.__aprob_history[i][j] = []
                self.__rho_history[i][j] = []
                self.__pure_regime[i][j] = []
                self.__unambi_regime_count[i].append(0)
        # --------------------------------------------------------------

    # API: methods for read only parameters
    @property
    def likelihood_history(self):
        return self.__likelihood_history

    @property
    def aprob_history(self):
        return self.__aprob_history

    @property
    def rho_history(self):
        return self.__rho_history

    @property
    def pure_regime(self):
        return self.__pure_regime

    @property
    def regime_shift_time(self):
        return self.__regime_shift_time

    def read_new_time(self, new_col):
        """
        *Callable*
        method that reads the new signal of the network,
        and then update the model likelihoods based on new signal.
        """

        # ----------------------------------------------------
        # deal with the first time instant
        # ----------------------------------------------------

        if self.__network_time == -1:

            self.__network_time = 0
            self.__new_regime_indicator = np.zeros(
                (self.__network_size, 1))
            for i in range(0, self.__network_size):
                self.__signal_history[i].append(0)
            self.__estimate_update()
            return 0

        # ----------------------------------------------------
        # after the first time instant
        # ----------------------------------------------------

        self.__network_time += 1
        self.__new_regime_indicator = np.copy(new_col)
        # update the regime history
        for i in range(0, self.__network_size):
            self.__signal_history[i].append(new_col[i])
        # update the prob of each of the model
        self.__estimate_update()
        return 0

    def __estimate_update(self):
        """
        Read the new signal for each new time instant.
        If there is a new regime for any of the nodes,
        we carry out the estimation algo.
        [core func]
        :return: the a posterior prob list for each node
        """
        # ----------------------------------------------------
        # deal with the first time instant
        # ----------------------------------------------------

        if self.__network_time == 0:
            prob_temp = 1 / 2
            for i in range(0, self.__network_size):
                range_temp = range(0, self.__network_size)
                range_temp.remove(i)
                for j in range(0, range_temp):
                    if j != i:
                        self.__likelihood_history[i][j].append(prob_temp)
                        self.__aprob_history[i][j].append(prob_temp)
            return 0

        # ----------------------------------------------------
        # after the first time instant
        # ----------------------------------------------------
        for i in range(0, self.__network_size):

            # i: the node of interest
            # j: the neighbor we want to assess if it's influencing node i

            # only start estimate if the current node starts a new regime
            # 1. check if this regime is pure
            # 2. if pure, then save the starting/ending time,
            #    the length, the index of the influencing node,
            #    the relative time of the influencing node
            #    (use a dict)
            if self.__new_regime_indicator[i] == 1:
                # list that saves the aprob of each model
                # a_prob_save = []

                # set of index of neighbors
                range_temp = range(0, self.__network_size)
                range_temp.remove(i)

                current_regime_start = self.__regime_shift_time[i][-1]
                current_regime_end = self.__network_time
                self.__regime_shift_time[i].append(current_regime_end)

                pure_ind = True
                total_influencing_node = 0
                influencing_node_index = None
                influencing_node_rep = 0
                influencing_node_time = []

                # go through all models to see if this new regime is pure
                for j in range_temp:
                    cond1 = \
                        current_regime_start < self.__regime_shift_time[j][-1]
                    cond2 = \
                        self.__regime_shift_time[j][-1] < current_regime_end
                    if cond1 and cond2:
                        influencing_node_index = j
                        total_influencing_node += 1
                        if total_influencing_node > 1:
                            pure_ind = False
                            break

                if not pure_ind:
                    continue

                if not influencing_node_index:
                    self.__pure_regime[i][i].append((current_regime_start,
                                                     current_regime_end))
                    influencing_node_index = i
                else:
                    for item in reversed(self.__signal_history[
                                             influencing_node_index]):
                        cond1 = current_regime_start < item
                        cond2 = item < current_regime_end
                        if cond1 and cond2:
                            influencing_node_rep += 1
                            influencing_node_time.append(item)
                        else:
                            break

                pure_regime_info = {"start_end": (current_regime_start,
                                                  current_regime_end),
                                    "length": current_regime_end -
                                                current_regime_start,
                                    "influencing_node": influencing_node_index,
                                    "influencing_rep": influencing_node_rep,
                                    "influencing_time": influencing_node_time}
                self.__pure_regime[i][influencing_node_index].append(
                    pure_regime_info)

                #### next: to 
                # go through all possible models and calculate the prob
                for j in range_temp:

                    # generate code for the current model
                    # (
                    # the ith node is located at the ith last digit
                    # if a digit is 1, that means the corresponding
                    # node is influencing the ith node in this model
                    # )

                    # current_model_code = self.__gen_code(i, j,
                    #                                     self.__network_size)

                    # calculate the likelihood of current model

                    # keep track of the number of "pure" regimes for the j^th
                    # model

                    model_likelihood, rho_est = self.__model_likelihood(i, j)
                    aprob_save.append(model_likelihood)

                    self.__likelihood_history[i][j].append(model_likelihood)
                    self.__rho_history[i][j].append(rho_est)

                # update the a posterior prob and save it, normalize

                normal_constant = sum(aprob_save)

                norm_aprob_save = [x / normal_constant for x in aprob_save]

                for j in range(0, 2 ** (self.__network_size - 1)):

                    self.__aprob_history[i][j].append(norm_aprob_save[j])

            # to make the program run faster we let the detection
            # only estimate in the last time slot
            # if self.__network_time == self.__total_time_instant - 1:

    def __model_likelihood(self, i, current_model):
        """
        Returns the likelihood of node *i* with *current_model*.
        Args:
            i: int
                The index of the node of interest.
            current_model:
                The model that we consider. [format?]
        Returns:
            model_prob: float in [0, 1]
                The calculated likelihood of *current_model*.
        """

        seq = self.__book_keeping_generic(i,
                                          self.__signal_history,
                                          current_model)

        alpha_est = self.__gibbs_sampling(seq)

        model_liklhd = self.__ys_seq_likelihood(seq, alpha_est)

        return model_liklhd, alpha_est

    def __book_keeping_generic(self, s_index, s_n, model):
        """
        Return the lengths of regimes given the model.
        This can be applied to any node over the network,
        with any model that is specified in the parameter.
        [Deterministic.]
        Args:
            s_index: int
                The index of the node of interest.
            s_n: dict
                The dict of signals history of all nodes
            model: binary string
                The hypothesis model.
                The value with the node of interest is always 0.
        Returns:
            n: 1d array
                The list of length of regimes of the node
                of interest.
        """

        s_obj = s_n[s_index]

        # obtain the index of influencing nodes by the current model
        neighbor_index_from_model = [self.__network_size-i-1
                                   for i, letter in enumerate(model)
                                   if letter == "1"]

        # merge the signals from all the influencing neighbors
        # so they can be looked as the equivalent of a single node

        new_sequence = [0. for j in range(0, len(s_obj))]

        for i in neighbor_index_from_model:

            for j in range(0, len(s_obj)):

                if s_n[i][j] == 1:
                    new_sequence[j] = 1

        book_keeping_results = self.__book_keeping_m1(new_sequence, s_obj)

        return book_keeping_results

    def __gibbs_sampling(self, n):

        # force the gibbs result to be 0.75
        # return 0.75

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

                    w = np.random.beta(a=alpha_e + 1, b=n[i], size=1)

                    b_draw_alpha = b_draw_alpha - log(w)

                else:

                    # i-YS case

                    w = np.random.beta(a=alpha_e, b=-n[i], size=1)

                    if w == 0:
                        w = 0.0000000000001

                    b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n)

            alpha_e = np.random.gamma(shape=a_draw_alpha, scale=1 / b_draw_alpha)

        return alpha_e

    @staticmethod
    def __book_keeping_m1(s1, s2):
        """
        Return the sequence of regime given M1.
        If the value in the returned list is negative, it means this regime ended
        because of a neighbor.
        Node 1 (s1) is what we want to decide if it has influence over node 2.
        Node 2 (s2) is the node of interest.
        This version is slightly different from the older version,
        in that it only append the length of the regime when it is finished.
        """

        n2 = np.array([])

        if len(s1) == len(s2):

            counter = 0

            for i in range(0, len(s1)):

                if s2[i] == 0 and s1[i] == 0:
                    counter += 1

                elif s2[i] == 1 and s1[i] == 0:
                    counter += 1
                    n2 = np.append(n2, counter)
                    counter = 0

                elif s2[i] == 0 and s1[i] == 1:
                    counter += 1
                    n2 = np.append(n2, -counter)
                    counter = 0

                elif s2[i] == 1 and s1[i] == 1:
                    counter += 1
                    n2 = np.append(n2, counter)
                    counter = 0

                else:
                    print("signal value is not 0 or 1.")
                    exit(-1)

        else:
            print("error!! len(s1) != len(s2)!! --function: book_keeping_m1")
            exit(-1)

        return n2

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

        for i in n:

            if i > 0:

                p *= alpha * scipy.special.beta(i, alpha + 1) * 1.5**abs(i)

            else:

                p *= alpha * scipy.special.beta(-i, alpha) * 1.5**abs(i)

        return p

    @staticmethod
    def __gen_code(i, j, network_size):
        # generate code for the current model

        # (
        # the ith node is located at the ith last digit
        # if a digit is 1, that means the corresponding
        # node is influencing the ith node in this model
        # )

        temp = bin(j)[2:].zfill(network_size - 1)

        if i == 0:
            current_model_code = temp + "0"

        else:
            current_model_code = ""

            for jj in range(0, len(temp)):

                if jj + i == network_size - 1:

                    current_model_code += "0" + temp[jj]

                else:

                    current_model_code += temp[jj]

        return current_model_code

