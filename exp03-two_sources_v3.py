# coding: utf-8
"""
Detect the i-YS process with two sources of influence.

Date: 05/31/2019
Author: Lingqing Gan @ Stony Brook University

Problem:
As in the following figure, node A is the object of interest.
We want to decide whether node B is influencing node A via i-YS;
Independent from this, we also want to decide whether node C is
influencing node A via i-YS.

   ╭───╮      ╭───╮
   | B |      | C |
   ╰─┬─╯      ╰─┬─╯
    i-YS?     i-YS?
     └───┐ ┌────┘
         ↓ ↓
        ╭┴─┴╮
        | A |
        ╰───╯

M0: node A is not i-YS influenced by node B or C
M1: node A is i-YS influenced by node B, but not node C
M2: node A is i-YS influenced by node C, but not node B
M3: node A is i-YS influenced by both node B and node C

Node A -> Node 0
Node B -> Node 1
Node C -> Node 2

Program Structure:
 * Signal generation
 * Model detection

Notes:
 * Start with deterministic model.


Ref:
 * legc10_YSP_inter_two_influence.py
 * legc02_YSP_inter_bi.py

Update 06/14/2019:
Finished writing the program. Now we start debugging it!!
First time writing program with object oriented programming..
It took some time. But I learned a lot.

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from functions.F02_IYSNetwork import IYSNetwork
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

    # =================================================
    #                      MODEL
    # =================================================

    # generate network topology
    adjacency_matrix = np.array([[0,0,0],[1,0,0],[1,0,0]])

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
        regime_detection.read_new_time(new_signal)

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

