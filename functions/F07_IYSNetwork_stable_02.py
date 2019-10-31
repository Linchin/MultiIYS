# coding: utf-8

"""
10/31/2019 note
stable_02
the code is examined and altered in accordance with F06-node stable
02, where we make the first signal 1 instead of 0.
10/24/2019 note
Rereading the code, checking the correctness. Improving the comments.

A class that contains the i-YS network.

F05 version:
this version is written to test if the detection code is correct.
The nodes are designed to generate some code that should yield
some simple and direct result.
The network class is changed to use these testing nodes.

"""
import numpy as np
from functions.F06_IYSDNode_stable_02 import IYSDNode_normal


class IYSNetwork:
    """
    A network where the nodes interact through i-YS.

    Attributes
    ----------
    __network_size : int
        Total number of nodes in the network.
    __adjacency_matrix : narray
        Adjacency matrix of the nodes, directed.
    __network_time : int
        The current time instant.
    __signal_history :
        The history of signals of all nodes.
    """

    def __init__(self, adjacency_matrix, rho = 0.75):
        """
        Creates an Interacting YS network object whose name is *name*,
        with a network decided by *adjacency_matrix*.
        :param adjacency_matrix : nparray
        """

        # instance variables (read only)
        self.__adjacency_matrix = adjacency_matrix
        self.__network_size = len(adjacency_matrix)
        self.__network_time = -1
        self.__rho = rho

        # history data (read only)
        self.__signal_history = {}
        for i in range(0, self.__network_size):
            self.__signal_history[i] = []

        # create the array of nodes and store it in instant variable
        self.__node_list = self.__create_nodes()

    # read only methods
    @property
    def adjacency_matrix(self):
        return self.__adjacency_matrix

    @property
    def network_size(self):
        return self.__network_size

    @property
    def signal_history(self):
        return self.__signal_history

    def __getitem__(self, item):
        """
        Make the nodes indexable.
        Return the node object indexed by *item*.
        """
        return self.__node_list[item]

    def __len__(self):
        """
        Know the number of nodes through len()
        Returns: int
        The number of nodes in the network.
        """
        return len(self.__node_list)

    def __create_nodes(self):
        """
        Create a node that is an object of the IYSNode class.
        A part of initiation.
        Returns: list
        The list of network nodes.
        """
        node_list = []
        for i in range(0, self.__network_size):
            # list of the indices of this node's neighbor
            # that are influencing the node
            node_neighbors = [
                j for j, x in enumerate(self.__adjacency_matrix[:, i])
                                                        if x == 1]
            node_list.append(IYSDNode_normal(
                        node_index=i,
                        node_neighbors=node_neighbors,
                        rho=self.__rho))
        return node_list

    def next_time_instant(self):
        """
        Callable method that brings the entire network to the next time instant.
        Returns: new_col
                 (array, size 1 x self.__network_size)
                 The array of new signals for the network.
        """
        self.__network_time += 1

        # new column of zeros that stores the new signals
        new_col = np.zeros(self.__network_size)

        # go through all the nodes
        for i in range(0, self.__network_size):
            # each node goes to new time instant
            new_signal = self[i].next_time_instant(self.__signal_history)
            # save the data in the history data list
            new_col[i] = new_signal
            self.__signal_history[i].append(new_signal)
        return new_col