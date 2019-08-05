# coding: utf-8

"""
To test the correctness of exp03, we implement a sure-converge example.

Date: 06/26/2019
Author: Lingqing Gan @ Stony Brook University

import:
F04_IYSNode
F05_IYSNetwork
F03_IYSDetection
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from functions.F05_IYSNetwork_test import IYSNetwork
from functions.F03_IYSDetection import IYSDetection


def main():
    """
    Main function.
    """

    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 2
    t = 500   # total number of time instants

    # =================================================
    #                      MODEL
    # =================================================



    # generate network topology
    # directed network.
    # item[i][j]=1 means node i influences node j
    # item[i][i]=0 all the time though each node technically
    # influences themselves
    adjacency_matrix = np.array([[0,0],[0,0]])

    # create the i-YS network object instance
    network = IYSNetwork(adjacency_matrix)

    # create the i-YS detection object instance
    regime_detection = IYSDetection(network_size)

    # for each time instant:
    for i in range(0, t):

        print("Current time: t=", i)

        # generate the network signal
        new_signal = network.next_time_instant()

        # run an online update
        regime_detection.read_new_time(np.copy(new_signal))

    # save the likelihood history
    aprob_history = regime_detection.aprob_history

    # =================================================
    #                      PLOT
    # =================================================

    # Create subplots
    # Set plot limits
    # fig: figure obj
    # axs: ndarray of sublot axis
    fig, axs = plt.subplots(nrows=network_size, ncols=1)

    # colors
    color_jet = cm.get_cmap('gist_rainbow')
    color_vector = np.linspace(0, 1, 2**(network_size-1))

    i = 0

    for ax in axs.reshape(-1):

        # Retrieve data for the current subplot
        current_node_history = aprob_history[i]

        # x-axis
        time_axis = [i for i in range(0, len(current_node_history[0]))]

        ax.set_xlim(min(time_axis), max(time_axis))
        ax.set_ylim(-0.03, 1.03)

        node_history_list = sorted(current_node_history.items())

        # cycle through all data line
        for item, c in zip(node_history_list, color_vector):

            l = item[0]
            j = item[1]

            current_color = color_jet(c)

            ax.plot(time_axis, j, color=current_color, label=l)

        ax.legend(loc="upper right")

        i += 1

    plt.show()

    print("Hey let's stop here.")

    return 0


if __name__ == "__main__":

    main()

