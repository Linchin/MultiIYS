"""
A class that detects the i-YS relationship.

12/02/2019 notes
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
        # Dict: __combined_signals
        # stores the combined unambiguous signal that we get after parsing.
        # i: the node of interest
        # j: the node likely to be the influencer.
        # Dict: __bookkeeping_results
        # stores the signal bookkeeping results
        # i: the node of interest
        # j: the node likely to be the influencer.

        self.__signal_history = {}      # type: Dict[int: List[Bool]]
        self.__likelihood_history = {}  # type: Dict[int: Dict[int: List[int]]]
        self.__aprob_history = {}       # type: Dict[int: Dict[int: List[int]]]
        self.__rho_history = {}         # type: Dict[int: Dict[int: List[tuple]]]
        self.__pure_regime = {}         # type: Dict[int: Dict[int: List[tuple]]]
        self.__regime_shift_time = {}   # type: Dict[int: List[int]]
        self.__unambi_regime_count = {} # type: Dict[int: List[int]]
        self.__ambi_regime_count = 0    # type: int
        self.__combined_signals = {}    # type: Dict[int: Dict[int: Dict["m0"/"m1": List]]]
        self.__bookkeeping_results = {}# type: Dict[int: Dict[int: Dict["m0"/"m1": array]]]
        for i in range(0, self.__network_size):
            self.__signal_history[i] = []
            self.__likelihood_history[i] = {}
            self.__aprob_history[i] = {}
            self.__rho_history[i] = {}
            self.__pure_regime[i] = {}
            self.__regime_shift_time[i] = []
            self.__unambi_regime_count[i] = []
            self.__combined_signals[i] = {}
            self.__bookkeeping_results[i] = {}
            for j in range(0, self.__network_size):
                self.__likelihood_history[i][j] = []
                self.__aprob_history[i][j] = []
                self.__rho_history[i][j] = []    # tuple (rho_aln, rho_ifld)
                self.__pure_regime[i][j] = []    # tuple (T, t)
                self.__unambi_regime_count[i].append(0)
                self.__combined_signals[i][j] = {} # type: Dict["m0"/"m1": List]
                self.__bookkeeping_results[i][j] = {} # type: Dict["m0"/"m1": List]

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

    @property
    def signal_history(self):
        return self.__signal_history

    @ property
    def combined_signals(self):
        return self.__combined_signals

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
            self.__new_regime_indicator = np.ones(
                (self.__network_size, 1))

            for i in range(0, self.__network_size):
                self.__signal_history[i].append(1)
                self.__regime_shift_time[i].append(0)
                self.__unambi_regime_count[i][i] += 1
                signals_temp_self = np.ones(2)
                self.__pure_regime[i][i].append(signals_temp_self)

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

            # special treatment for the last time instant
            if new_col[i] == 0:
                continue

            self.__regime_shift_time[i].append(self.__network_time)
            # decide if this new regime is ambiguous or unambiguous
            begin = self.__regime_shift_time[i][-2]
            end = self.__regime_shift_time[i][-1]
            count_parse = 0

            for j in range(0, self.__network_size):
                if j == i:
                    continue
                last_rgm_shft = self.__regime_shift_time[j][-1]
                if last_rgm_shft == end:
                    last_rgm_shft = self.__regime_shift_time[j][-2]
                if begin < last_rgm_shft <= end-1:
                    count_parse += 1
                    influencer = j

            # case 1: there is no influencing neighbor
            if count_parse == 0:
                self.__unambi_regime_count[i][i] += 1
                # reconstruct the self influencing signals
                signals_temp_self = np.zeros(end-begin+1)
                signals_temp_self[0] = 1
                signals_temp_self[-1] = 1
                self.__pure_regime[i][i].append(signals_temp_self)

            # case 2: there is exactly 1 possible influencer
            elif count_parse == 1:
                self.__unambi_regime_count[i][influencer] += 1
                # reconstruct the signals
                regime_recon = np.zeros(end-begin+1)      # influenced node
                regime_recon[0] = 1
                regime_recon[-1] = 1
                relative_inf_time = []  # include all relative influencing time
                # find out the list of relative influence time
                for item in self.__regime_shift_time[influencer]:
                    if begin < item <= end-1:
                        relative_inf_time.append(item-begin)
                self.__pure_regime[i][influencer].append((relative_inf_time, regime_recon))
                # length of the current regime; relative time point of
                # the possible influence

            # case 3: there are at least 2 possible influencers
            else:
                self.__ambi_regime_count += 1

        # print the number of unambiguous regimes for each node
        # and possible influencing nodes
        # can i print it as a matrix??
        stdout.write("Total regimes: %d; \nTotal ambiguous regimes: %d; \n"
                     % (self.__network_time,
                        self.__ambi_regime_count))
        print_string = "Unambiguous regimes #+ of node %d: " + \
                       "%d, " * (self.__network_size-1) + "%d" + "\n"
        for i_temp in range(0, self.__network_size):
            print_data = tuple([i_temp]) + tuple(self.__unambi_regime_count[i_temp][j]
                               for j in range(0,self.__network_size))
            #print()
            stdout.write(print_string % print_data)



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
        # skip estimation if not yet at last time instant:
        if self.__network_time + 1 != self.__total_time_instant:
            return 0
        # ----------------------------------------------------
        # deal with the first time instant
        # ----------------------------------------------------
        # if self.__network_time == 0:
        #     prob_temp = (1/2, 1/2)
        #     none_double = (None, None)
        #     for i in range(0, self.__network_size):
        #         for j in range(0, self.__network_size):
        #             if j != i:
        #                 self.__likelihood_history[i][j].append(prob_temp)
        #                 self.__aprob_history[i][j].append(prob_temp)
        #             else:
        #                 self.__likelihood_history[i][j].append(none_double)
        #                 self.__aprob_history[i][j].append(none_double)
        #     return 0
        # ----------------------------------------------------
        # 2nd time instant and later
        # ----------------------------------------------------
        for i in range(0, self.__network_size):
            for j in range(0, self.__network_size):

                """
                Case 1: self influencing, then skip, no need to estimate
                """
                if i == j:
                    continue

                """
                Case 2: decide if node j influences node i
                """
                # 1) Generate the sequence related to node i & j
                #    given influenced/not influenced
                s_sf = self.__pure_regime[i][i]
                s_nb = self.__pure_regime[i][j]

                s_combined_m0 = self.__combine_parsed_signals(s_sf, s_nb, "m0")
                s_combined_m1 = self.__combine_parsed_signals(s_sf, s_nb, "m1")

                n_m0 = self.__book_keeping_from_time(s_combined_m0)
                n_m1 = self.__book_keeping_from_time(s_combined_m1)

                # 2) Gibbs sampling on the just generated sequence
                alpha_m0 = self.__gibbs_sampling(n_m0)
                alpha_m1 = self.__gibbs_sampling(n_m1)

                # 3) Calculate the likelihood of both models
                lklhd_m0 = self.__ys_seq_likelihood(n_m0, alpha_m0)
                lklhd_m1 = self.__ys_seq_likelihood(n_m1, alpha_m1)
                print(" Node of interest:", i, " Influencing node:", j,
                      " P(M_0):", lklhd_m0, " P(M_1):", lklhd_m1)

                # 4) Model selection
                temp_norm = lklhd_m0 + lklhd_m1
                aprob_m0, aprob_m1 = lklhd_m0 / temp_norm, lklhd_m1 / temp_norm

                # 5) Save the data
                self.__likelihood_history[i][j].append((lklhd_m0, lklhd_m1))
                self.__aprob_history[i][j].append((aprob_m0, aprob_m1))
                self.__rho_history[i][j].append((alpha_m0, alpha_m1))
                self.__combined_signals[i][j]["m0"] = s_combined_m0
                self.__combined_signals[i][j]["m1"] = s_combined_m1
                self.__bookkeeping_results[i][j]["m0"] = n_m0
                self.__bookkeeping_results[i][j]["m1"] = n_m1

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
                    w = np.random.beta(a=alpha_e+1, b=n[i], size=1)
                    b_draw_alpha = b_draw_alpha - log(w)
                else:
                    # i-YS case
                    w = np.random.beta(a=alpha_e, b=-n[i], size=1)
                    if w == 0:
                        w = 0.0000000000001
                    b_draw_alpha = b_draw_alpha - log(w)

            a_draw_alpha = a_e + len(n)
            alpha_e = np.random.gamma(shape=a_draw_alpha, scale=1/b_draw_alpha)

        return alpha_e

    def __combine_parsed_signals(self, s_sf, s_nb, hypo):
        """
        s_sf: the collected unambiguous regimes with no possible influencers
        s_nb: the collected unambiguous regimes with one possible influencer
        hypo: "m0" for not influenced, "m1" for influenced.

        Returns the reconstructed signals from the parsed signals based on
        hypothesis.

        """
        if hypo == "m0" or hypo == "m1":
            if len(s_sf) == 0 and len(s_nb) == 0:
                return np.zeros(0)

            # initialize the combined signal
            s_combined = np.ones(0)

            # self (regimes with no possible influencer)
            for i in range(0, len(s_sf)):
                if len(s_sf[i]) > 1:
                    temp = np.zeros(len(s_sf[i]) - 1)
                    temp[-1] = 1
                    s_combined = np.concatenate((s_combined, temp))
                else:
                    print(i, s_sf[i])
                    print("Error!! bookkeeping m0 signal length abnormal.")
                    exit(-1)

            # from neighbor
            for i in range(0, len(s_nb)):
                temp = np.zeros(len(s_nb[i][1]) - 1)
                temp[-1] = 1
                if hypo == "m1":            # mark it as -1
                    for item in s_nb[i][0]:
                        # change all influenced signals
                        # to -1 in the current regime
                        # 11/07/2019 change
                        temp[item - 1] = -1
                s_combined = np.concatenate((s_combined, temp))

            return s_combined

        else:
            print("Function __combine_parsed_signals wrong hypothesis code.")
            exit(-1)


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
                p *= alpha * scipy.special.beta(-i_l, alpha)
        return p


