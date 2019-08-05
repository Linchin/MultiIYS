"""
Class for i-YS network nodes.
"""

import numpy as np


class IYSDNode:
    """
    (Interactive Yule Simon Deterministic Node)
    A node in the network that follows DETERMINISTIC interacting
    YS (i-YS) processes.

    Description
    -----------
    For each time instant:
    IF state_signal = 1 for any of {this node, this node's neighbors}:
        counter <= 1;
    ELSE
        counter <= counter + 1.
    state_signal <= Bernoulli[rho/(counter+rho)]

    Attributes
    ----------
    __node_index : int
        Index of the node in the network.
    __node_neighbors : list
        Indices of this node's neighbors.
    __time_instant : int [1, ∞)
        Time instant this node is at right now.
    __counter : int
        Counter for the YS decision.
    __rho : float > 0
        Value of rho parameter.
    __regime_history: list (int)
        List that stores the length of each finished regime.
    __signal_history: list (int)
        List that stores the past signals.

    """

    def __init__(self,
                 node_index,
                 node_neighbors,
                 rho=0.5):
        """Return an InteractingYS_Node object whose index is
        *node_index*, and rho is *rho*. *node_neighbors* is a
        list of the indices of the neighbors.
        """

        # constant parameters read only
        self.__node_index = node_index
        self.__node_neighbors = node_neighbors
        self.__rho = rho

        # parameters that will change over time
        self.__time_instant = -1
        self.__counter = 0

        # list that stores the history
        # length of each *FINISHED* regime
        self.__regime_history = []
        self.__signal_history = []

    # create methods for read only parameters
    @property
    def index(self):
        return self.__node_index

    @property
    def neighbors(self):
        return self.__node_neighbors

    @property
    def rho(self):
        return self.__rho

    @property
    def time(self):
        return self.__time_instant

    @property
    def counter(self):
        return self.__counter

    @property
    def regime_history(self):
        return self.__regime_history

    @property
    def signal_history(self):
        return self.__signal_history

    def next_time_instant(self, network_signal_history):
        """
        Bring this node to the next time instant.
        Args:
            network_signal_history : dict
        Returns:
            new_signal : int, 0 or 1
                Returns the new signal of this node.
        """

        # the starting time instant
        if self.__time_instant == -1:

            self.__time_instant = 0
            new_signal = 0
            self.__counter += 1
            self.__signal_history.append(new_signal)

            return new_signal

        else:

            old_time = self.__time_instant

            self.__time_instant += 1

            # decide if this node will start a new regime

            prob = self.__rho / (self.__counter + self.__rho)

            new_signal = np.random.binomial(1, prob)

            # decide if the node will reset its counter
            # combined signals from neighbors' last observation
            combined_signal = 0

            # deterministic
            if len(self.__node_neighbors)>0:

                for neighbor_index in self.__node_neighbors:

                    neighbor_sig = network_signal_history[neighbor_index]\
                    [old_time]

                    combined_signal = combined_signal or neighbor_sig

            # update the counter value
            if new_signal == 1:

                self.__regime_history.append(self.__counter)
                self.__counter = 1

            elif new_signal == 0:

                if combined_signal == 0:
                    self.__counter += 1
                elif combined_signal == 1:
                    self.__regime_history.append(self.__counter)
                    self.__counter = 1
                else:
                    print("signal value error.")

            else:
                print("new signal error - node, next time instant")

            # update the corresponding data

            self.__signal_history.append(new_signal)

            return new_signal


