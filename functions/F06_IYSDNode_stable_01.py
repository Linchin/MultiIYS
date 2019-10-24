# coding: utf-8

"""
10/24/2019 note
reread. improve comments. for the parsed case.

Class for i-YS network nodes.

F04 version:
this version is written to test if the detection code is correct.
The nodes are designed to generate some code that should yield
some simple and direct result.
"""
import numpy as np


class IYSDNode_normal:
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
        Indices of this node's neighbors. that will influence this node.
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
                 rho=0.75):
        """Return an InteractingYS_Node object whose index is
        *node_index*, and rho is *rho*. *node_neighbors* is a
        list of the indices of the neighbors.
        """

        # constant parameters read only
        self.__node_index = node_index
        self.__node_neighbors = node_neighbors.copy()
        self.__rho = rho

        # parameters that will change over time
        self.__time_instant = -1
        self.__counter = 0

        # signal history of this node
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

        if self.__time_instant == -1:
            # 1st time instant
            self.__time_instant = 0
            new_signal = 0
            self.__counter = 1
            self.__signal_history.append(new_signal)
            return new_signal

        else:
            # 2nd time instant and later
            old_time = self.__time_instant
            self.__time_instant += 1
            # decide if the node will reset its counter
            # DETERMINISTIC NODE
            # combine the neighbor signals
            combined_signal = 0
            if len(self.__node_neighbors) > 0:
                for neighbor_index in self.__node_neighbors:
                    neighbor_sig = network_signal_history[
                        neighbor_index][old_time]
                    if neighbor_sig == 1:
                        combined_signal = 1

            # update the counter value
            last_signal = self.__signal_history[-1]

            if last_signal == 1:
                self.__counter = 1
            elif last_signal == 0:
                if combined_signal == 0:
                    self.__counter += 1
                elif combined_signal == 1:
                    self.__counter = 1
                else:
                    print("combined neighbor signal value error.")
            else:
                print("new signal error - node class, next time instant")

            # decide if this node will start a new regime
            prob = self.__rho / (self.__counter + self.__rho)
            new_signal = np.random.binomial(1, prob)
            self.__signal_history.append(new_signal)
            # update the corresponding data
            return new_signal
