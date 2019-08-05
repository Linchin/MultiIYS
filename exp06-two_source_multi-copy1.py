# coding: utf-8

"""
Date: 07/08/2019
Author: Lingqing Gan @ Stony Brook University

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
    network_size = 3
    t = 1000   # total number of time instants

    total_rep = 100

    aprob_best_model_hist = {}

    for i in range(0, network_size):

        aprob_best_model_hist[i] = []


    for rep_exp_index in range(0, total_rep):

        print("Current repetition: rep=", rep_exp_index)

        # =================================================
        #                      MODEL
        # =================================================

        # generate network topology
        # directed network.
        # item[i][j]=1 means node i influences node j
        # item[i][i]=0 all the time though each node technically
        # influences themselves
        adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])

        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix)

        # create the i-YS detection object instance
        regime_detection = IYSDetection(network_size)

        # for each time instant:
        for i in range(0, t):

#            print(i)

            if i == 0:
                print("Start of current rep.")
            elif i == t-1:
                print("Start of Gibbs sampling...")

            # generate the network signal
            new_signal = network.next_time_instant()

            # run an online update
            regime_detection.read_new_time(np.copy(new_signal))

            if i == t-1:
                print("End of Gibbs sampling.")

        # save the likelihood history
        aprob_history = regime_detection.aprob_history

        for i in range(0, network_size):

            final_aprob = []

            for j in range(0, 2**(network_size-1)):

                final_aprob.append(aprob_history[i][j][-1])

            max_model_index = final_aprob.index(max(final_aprob))

            aprob_best_model_hist[i].append(max_model_index)

    print(aprob_best_model_hist)


    #
    #
    # # =================================================
    # #                      PLOT
    # # =================================================
    #
    # # Create subplots
    # # Set plot limits
    # # fig: figure obj
    # # axs: ndarray of sublot axis
    # fig, axs = plt.subplots(nrows=network_size, ncols=1)
    #
    # # colors
    # color_jet = cm.get_cmap('gist_rainbow')
    # color_vector = np.linspace(0, 1, 2**(network_size-1))
    #
    # i = 0
    #
    # for ax in axs.reshape(-1):
    #
    #     # Retrieve data for the current subplot
    #     current_node_history = aprob_history[i]
    #
    #     # x-axis
    #     time_axis = [i for i in range(0, len(current_node_history[0]))]
    #
    #     ax.set_xlim(min(time_axis), max(time_axis))
    #     ax.set_ylim(-0.03, 1.03)
    #
    #     node_history_list = sorted(current_node_history.items())
    #
    #     # cycle through all data line
    #     for item, c in zip(node_history_list, color_vector):
    #
    #         l = item[0]
    #         j = item[1]
    #
    #         current_color = color_jet(c)
    #
    #         ax.plot(time_axis, j, color=current_color, label=l)
    #
    #     ax.legend(loc="upper right")
    #
    #     i += 1
    #
    # plt.show()
    #
    # print("Hey let's stop here.")

    return 0


if __name__ == "__main__":

    main()




