"""
A class that detects the i-YS relationship.

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


class IYSDetection_parse:
    """
    Detect the i-YS relationship on the network.
    We parse the sequence so we only use the unambiguous sequence for
    detection.
    """

    def __init__(self, network_size, gibbs_rep, time_inst):
        """
        Create an i-YS relationship detection object.
        """
        # --------------------------------------------------------------
        # 1. Parameters set up
        # --------------------------------------------------------------
        self.__rep_alpha = gibbs_rep  # rounds of Gibbs sampler for alpha
        self.__network_size = network_size  # number of nodes in the network

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
        # 2. Construct data structures to store all necessary history data
        # ----------------------------------------------------------------
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
        self.__signal_history = {}      # type: Dict[int: List[Bool]]
        self.__likelihood_history = {}  # type: Dict[int: Dict[int: List[int]]]
        self.__aprob_history = {}       # type: Dict[int: Dict[int: List[int]]]
        self.__rho_history = {}         # type: Dict[int: Dict[int: List[tuple]]]
        self.__pure_regime = {}         # type: Dict[int: Dict[int: List[tuple]]]
        self.__regime_shift_time = {}   # type: Dict[int: List[int]]
        self.__unambi_regime_count = {} # type: Dict[int: List[int]]
        self.__ambi_regime_count = 0    # type: int
        for i in range(0, self.__network_size):
            self.__signal_history[i] = []
            self.__likelihood_history[i] = {}
            self.__aprob_history[i] = {}
            self.__rho_history[i] = {}
            self.__pure_regime[i] = {}
            self.__regime_shift_time[i] = []
            self.__unambi_regime_count[i] = []
            for j in range(0, self.__network_size):
                self.__likelihood_history[i][j] = []
                self.__aprob_history[i][j] = []
                self.__rho_history[i][j] = []    # tuple (rho_aln, rho_ifld)
                self.__pure_regime[i][j] = []    # tuple (T, t)
                self.__unambi_regime_count[i].append(0)

    # ----------------------------------------------------------------
    # API: methods for read only parameters
    # ----------------------------------------------------------------
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
        method that
        1) reads the new signal of the network,
        2) parse the data,
        3) save the data to the structs in this object.
        4) call self.__estimate_update() to update the model likelihood.
        """
        # ----------------------------------------------------
        # 1st time instant
        # ----------------------------------------------------
        if self.__network_time == -1:
            self.__network_time = 0
            self.__new_regime_indicator = np.zeros(
                (self.__network_size, 1))
            for i in range(0, self.__network_size):
                self.__signal_history[i].append(0)
                self.__regime_shift_time[i].append(0)
            self.__estimate_update()
            return 0
        # ----------------------------------------------------
        # 2nd time instant and later
        # ----------------------------------------------------
        self.__network_time += 1
        self.__new_regime_indicator = np.copy(new_col)
        # update the regime history
        for i in range(0, self.__network_size):
            self.__signal_history[i].append(new_col[i])
            # for any node that has started a new regime, we update the
            # following data structure
            if new_col[i] == 1:
                self.__regime_shift_time[i].append(self.__network_time)
                # decide if this new regime is ambiguous or unambiguous
                begin = self.__regime_shift_time[i][-2]
                end = self.__regime_shift_time[i][-1]
                count = 0
                for j in range(0, self.__network_size):
                    if j == i:
                        continue
                    last_rgm_shft = self.__regime_shift_time[j][-1]
                    if last_rgm_shft == end:
                        if len(self.__regime_shift_time[j]) >= 2:
                            last_rgm_shft = self.__regime_shift_time[j][-2]
                        else:
                            continue
                    if begin <= last_rgm_shft < end-1: # added -1 for the delay
                        count += 1
                        influencer = j
                        inf_time = last_rgm_shft
                # case 1: there is no influencing neighbor
                if count == 0:
                    self.__unambi_regime_count[i][i] += 1
                    self.__pure_regime[i][i].append((end-begin, None))
                # case 2: there is exactly 1 possible influencer
                elif count == 1:
                    self.__unambi_regime_count[i][influencer] += 1
                    self.__pure_regime[i][influencer].append((end-begin,
                                                              inf_time-begin))
                    # length of the current regime; relative time point of
                    # the possible influence
                # case 3: there are at least two possible influencers
                else:
                    self.__ambi_regime_count += 1

        # print the current number of each type of regimes.
        # stdout.write("\r%d" % i)
        # stdout.flush()
        # stdout.write("Total: %d; Ambi: %d; Unam: %d %d %d %d %d %d %d %d %d\n"
        #              % (self.__network_time,
        #                 self.__ambi_regime_count,
        #                 self.__unambi_regime_count[0][0],
        #                 self.__unambi_regime_count[0][1],
        #                 self.__unambi_regime_count[0][2],
        #                 self.__unambi_regime_count[1][0],
        #                 self.__unambi_regime_count[1][1],
        #                 self.__unambi_regime_count[1][2],
        #                 self.__unambi_regime_count[2][0],
        #                 self.__unambi_regime_count[2][1],
        #                 self.__unambi_regime_count[2][2]
        #                 ))

        stdout.write("Total: %d; Ambi: %d; Unam: %d %d %d %d\n"
                     % (self.__network_time,
                        self.__ambi_regime_count,
                        self.__unambi_regime_count[0][0],
                        self.__unambi_regime_count[0][1],
                        self.__unambi_regime_count[1][0],
                        self.__unambi_regime_count[1][1]
                        ))
        # stdout.flush()
        # update the prob of each of the model
        self.__estimate_update()
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
        # ----------------------------------------------------
        # deal with the first time instant
        # ----------------------------------------------------
        if self.__network_time == 0:
            prob_temp = (1/2, 1/2)
            none_double = (None, None)
            for i in range(0, self.__network_size):
                for j in range(0, self.__network_size):
                    if j != i:
                        self.__likelihood_history[i][j].append(prob_temp)
                        self.__aprob_history[i][j].append(prob_temp)
                    else:
                        self.__likelihood_history[i][j].append(none_double)
                        self.__aprob_history[i][j].append(none_double)
            return 0
        # ----------------------------------------------------
        # 2nd time instant and later
        # ----------------------------------------------------
        # skip estimation if not yet at last time instant:
        if self.__network_time + 1 != self.__total_time_instant:
            return 0
        for i in range(0, self.__network_size):
            for j in range(0, self.__network_size):
                # Case 1: self influencing, then skip
                if i == j:
                    prob_temp = (1/2, 1/2)
                    none_double = (None, None)
                    self.__likelihood_history[i][j].append(prob_temp)
                    self.__aprob_history[i][j].append(prob_temp)
                    self.__rho_history[i][j].append(none_double)
                    continue
                # Case 2: exactly 1 influencer
                # 1) Generate the sequence related to node i & j
                s_sf = self.__pure_regime[i][i]
                s_nb = self.__pure_regime[i][j]
                n_aln = self.__book_keeping_m0_from_time(s_sf, s_nb)
                n_ifcd = self.__book_keeping_m1_from_time(s_sf, s_nb)
                # 2) Gibbs sampling on the just generated sequence
                alpha_aln = self.__gibbs_sampling(n_aln)
                alpha_ifcd = self.__gibbs_sampling(n_ifcd)
                # 3) Calculate the likelihood of both models
                lklhd_aln = self.__ys_seq_likelihood(n_aln, alpha_aln)
                lklhd_ifcd = self.__ys_seq_likelihood(n_ifcd, alpha_ifcd)
                sum_aln = 0
                sum_ifcd = 0
                for item in n_aln:
                    sum_aln += abs(item)
                for item in n_ifcd:
                    sum_ifcd += abs(item)
                print(len(n_aln), sum_aln, n_aln)
                print(len(n_ifcd), sum_ifcd, n_ifcd)
                print(i, j, lklhd_aln, lklhd_ifcd)
                # 4) Model selection
                temp = lklhd_aln + lklhd_ifcd
                aprob_aln, aprob_ifcd = lklhd_aln / temp, lklhd_ifcd / temp
                # 5) Save the data
                self.__likelihood_history[i][j].append((lklhd_aln, lklhd_ifcd))
                self.__aprob_history[i][j].append((aprob_aln, aprob_ifcd))
                self.__rho_history[i][j].append((alpha_aln, alpha_ifcd))

    def __gibbs_sampling(self, n):
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
                    # print(alpha_e, n[i])
                    w = np.random.beta(a=alpha_e, b=-n[i], size=1)
                    if w == 0:
                        w = 0.0000000000001
                    b_draw_alpha = b_draw_alpha - log(w)
            a_draw_alpha = a_e + len(n)
            alpha_e = np.random.gamma(shape=a_draw_alpha, scale=1/b_draw_alpha)
        return alpha_e

    @staticmethod
    def __book_keeping_m1_from_time(s_sf, s_nb):
        """
        Return the bookkeeping sequence given the hypothesis that the neighbor
        has influence.
        s_sf: the list of regimes where the node is not influenced by anyone
              (self)
        s_nb: the list of regimes where the node is possible to be influenced
              by one neighbor.
              (neighbor)
        :return: n
        """
        n = []
        for item in s_sf:
            n.append(item[0])
        for item in s_nb:
            n.append(-item[1]-1) # notes: -1 to change the counting
            n.append(item[0]-item[1]-1) # notes: -1 to change the counting

        return n

    @staticmethod
    def __book_keeping_m0_from_time(s_sf, s_nb):
        """
        Return the bookkeeping sequence given the hypothesis that the neighbor
        has no influence.
        s_sf: the list of regimes where the node is not influenced by anyone
              (self)
        s_nb: the list of regimes where the node is possible to be influenced
              by one neighbor.
              (neighbor)
        :return: n
        """
        n = []
        # should I keep this or not??
        for item in s_sf:
            n.append(item[0])
        for item in s_nb:
            n.append(item[0])
        return n

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
                p *= alpha * scipy.special.beta(i, alpha + 1)
            else:
                p *= alpha * scipy.special.beta(-i, alpha)
        return p


