"""
A class that detects the i-YS relationship.
08/12/2019
added the history of estimated rho value.
An online version.
Deterministic.

Des:
1. Based on F08, the stable detection;
2. Altered to do online detection;
3. Remove the branch when the prob is lower than a given threshold.
"""

from math import log

import numpy as np
import scipy.special


class IYSDetection:
    """
    Detect the i-YS relationship on the network.
    """

    def __init__(self,
                 network_size,
                 gibbs_rep,
                 threshold=0.1):
        """
        Create an i-YS relationship detection object.
        """

        # parameters

        self.__rep_alpha = gibbs_rep  # rounds of Gibbs sampler for alpha
        self.__network_size = network_size
        self.__threshold = threshold

        # signal history of the entire network
        self.__signal_history = {}
        for i in range(0, self.__network_size):
            self.__signal_history[i] = []

        # indicator of starting new regime at the new time instant
        self.__new_regime_indicator = np.zeros((self.__network_size, 1))

        # time instant starting from 0
        self.__network_time = -1

        # store the likelihood history of each model
        # store the a posterior prob history of each model
        # store the gibbs sampling results history of each model
        # store the viable paths (model) for each node

        self.__likelihood_history = {}
        self.__aprob_history = {}
        self.__rho_history = {}
        self.__viable_paths = {}

        for i in range(0, self.__network_size):

            self.__likelihood_history[i] = {}
            self.__aprob_history[i] = {}
            self.__rho_history[i] = {}
            self.__viable_paths[i] = {}

            for j in range(0, 2**(self.__network_size-1)):

                self.__likelihood_history[i][j] = []
                self.__aprob_history[i][j] = []
                self.__rho_history[i][j] = []

    # methods for read only parameters
    @property
    def likelihood_history(self):
        return self.__likelihood_history

    @property
    def aprob_history(self):
        return self.__aprob_history

    @property
    def rho_history(self):
        return self.__rho_history

    def read_new_time(self, new_col):
        """
        *Callable* method that reads the new signal of the network,
        and then update the model likelihoods based on new signal.
        Args:
            new_col: the new column of signals

        Returns:
            None.
        """

        # ----------------------------------------------------
        # deal with the 1st time instant
        # ----------------------------------------------------

        if self.__network_time == -1:

            self.__network_time = 0
            self.__new_regime_indicator = np.zeros((self.__network_size, 1))

            for i in range(0, self.__network_size):

                self.__signal_history[i].append(0)

                for j in range(0, 2 ** (self.__network_size-1)):

                    path_code = self.__model_code_gen(i, j)
                    self.__viable_paths[i][j] = path_code

            # update the prob of each of the model
            self.__estimate_update()

            return 0

        # ----------------------------------------------------
        # after the 1st time instant
        # ----------------------------------------------------

        self.__network_time += 1

        self.__new_regime_indicator = np.copy(new_col)

        # update the regime history
        for i in range(0, self.__network_size):

            # append the new signals to the history
            self.__signal_history[i].append(new_col[i])

        # update the prob of each of the model
        self.__estimate_update()

        # no need to update the viable paths in the first time
        # instant since it is set up in __init__()

        return 0

    def __estimate_update(self):
        """
        Internal function.
        Read the new signal for each new time instant.
        If there is a new regime for any of the nodes,
        we carry out the estimation algo.
        [core func]
        :return: the a posterior prob list for each node
        """

        # ----------------------------------------------------
        # deal with the 1st time instant
        # ----------------------------------------------------

        if self.__network_time == 0:

            prob_temp = 1 / 2 ** (self.__network_size - 1)

            for i in range(0, self.__network_size):

                for j in range(0, 2 ** (self.__network_size - 1)):

                    path_code = self.__model_code_gen(i, j)

                    self.__likelihood_history[i][j].append(prob_temp)
                    self.__aprob_history[i][j].append(prob_temp)
                    self.__viable_paths[i][j] = path_code

            return 0

        # ----------------------------------------------------
        # after the 1st time instant
        # ====================================================
        # Change for F09:
        # 1. the update is online;
        # 2. remove the trace and the following up traces;
        #
        # ----------------------------------------------------

        for i in range(0, self.__network_size):

            # only start estimate if the current node starts a new regime
            if self.__new_regime_indicator[i] == 1:
            #if self.__network_time == 999:

                #*****************
                # function: add trace
                #*****************
                #  self.__add_trace()

                # list that saves the aprob of each model
                aprob_save = []

                # go through all possible models and calculate the prob
                # !!! NEEDS CHANGE INTO A SET OF VIABLE TRACES
                for j in range(0, 2**(self.__network_size-1)):

                    print(i,j)

                    # # generate code for the current model

                    # # the ith node is located at the ith digit from last
                    # # if a digit is 1, that means the corresponding
                    # # node is influencing the ith node in this model

                    # # generate the code w.r.t. all neighbors
                    # temp = bin(j)[2:].zfill(self.__network_size-1)
                    #
                    # # insert a "0" in the spot for the ith node
                    # if i == 0:
                    #     current_model_code = temp + "0"
                    #
                    # else:
                    #     current_model_code = ""
                    #
                    #     for jj in range(0, len(temp)):
                    #
                    #         if jj + i == self.__network_size - 1:
                    #
                    #             current_model_code += "0" + temp[jj]
                    #
                    #         else:
                    #
                    #             current_model_code += temp[jj]

                    current_model_code = self.__model_code_gen(i, j)

                    # calculate the likelihood of current model

                    model_likelihood, rho_est = self.__model_likelihood_cut_traces(
                                                       i, j, current_model_code)

                    print(model_likelihood)
                    aprob_save.append(model_likelihood)

                    # save the likelihood, rho, aprob to history data
                    self.__likelihood_history[i][j].append(model_likelihood)
                    self.__rho_history[i][j].append(rho_est)

                # update the a posterior prob and save it

                normal_constant = sum(aprob_save)

                norm_aprob_save = [x / normal_constant for x in aprob_save]

                for j in range(0, 2 ** (self.__network_size - 1)):

                    self.__aprob_history[i][j].append(norm_aprob_save[j])

                # function: remove traces

                self.__remove_trace(i)


    def __model_code_gen(self, i, j):
        """
        Generate code for the current model.
        Parameter:
        i: the index of the current node
        j: the index of the model
        Return:
        a code for the current model as a string.
        Notes:
         * the ith node is located at the ith digit from last;
         * if a digit is 1, that means the corresponding
           node is influencing the ith node in this model.
        """

        # generate the code w.r.t. all neighbors
        temp = bin(j)[2:].zfill(self.__network_size - 1)

        # insert a "0" in the spot for the ith node
        if i == 0:
            current_model_code = temp + "0"

        else:
            current_model_code = ""

            for jj in range(0, len(temp)):

                if jj + i == self.__network_size - 1:

                    current_model_code += "0" + temp[jj]

                else:

                    current_model_code += temp[jj]

        return current_model_code



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

    def __model_likelihood_cut_traces(self, i, j, current_model):
        """
        (altered version of __model_likelihood())
        Returns the likelihood of node *i* with *current_model*.
        Update:
        if this code is in the viable set then proceed
        otherwise assign aprob value and est rho to 0
        Also the estimated rho value becomes -0.1

        Args:
            i: int
                The index of the node of interest.
            current_model:
                The model that we consider. [format?]
        Returns:
            model_prob: float in [0, 1]
                The calculated likelihood of *current_model*.
        """

        if j in self.__viable_paths:

            seq = self.__book_keeping_generic(i,
                                              self.__signal_history,
                                              current_model)

            rho_est = self.__gibbs_sampling(seq)

            model_liklhd = self.__ys_seq_likelihood(seq, rho_est)

            return model_liklhd, rho_est

        else:

            return 0, -0.1


#    def __add_trace(self):
#        """
#        When there starts a new regime, update self.__viable_paths
#        """


    def __remove_trace(self, i):
        """
        After calculating the prob of the viable paths, remove the
        paths with aprob < threshold.
        """

        for j in list(self.__viable_paths[i]):

            if self.__aprob_history[i][j][-1] < self.__threshold:

                del self.__viable_paths[i][j]


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

        # obtain the index of influential nodes by the current model
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

            alpha_e = np.random.gamma(shape=a_draw_alpha,
                                      scale=1 / b_draw_alpha)

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

