# coding: utf-8

"""
Date: 08/12/2019
Author: Lingqing Gan @ Stony Brook University

Title: exp11-online update.py

Description:
1. Adapted from exp07;
2. Single experiment, just let total_rep=1;
3. Online update;
4. Remove the branch when the prob is lower than a given threshold.
5. Major detection change lies in F09.
6. If a possible trace is removed, the prob just saves as -0.1;
"""

import numpy as np
import time
import pickle

from functions.F07_IYSNetwork_stable_01 import IYSNetwork
from functions.F09_IYSDetection_online_efficient_01 import IYSDetection

def main():
    """
    Main function.
    """

    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 3
    t = 1000   # total number of time instants

    threshold = 0.1 # 0 < threshold < 1

    total_rep = 10

    gibbs_rep = 10000

    rho = 0.75

    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    aprob_best_model_hist = {}

    # data section of the dict to save
    data_dict = {}

    for i in range(0, network_size):

        aprob_best_model_hist[i] = []

        data_dict[i] = {}

        for j in range(0, 2**(network_size-1)):

            data_dict[i][j] = {"rho": [],
                               "aprob": []}

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
        adjacency_matrix = np.array([[0,0,0], [0,0,0], [0,0,0]])

        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix, rho=rho)

        # create the i-YS detection object instance
        regime_detection = IYSDetection(network_size=network_size,
                                        gibbs_rep=gibbs_rep,
                                        threshold=threshold)

        # for each time instant:
        for i in range(0, t):

            # generate the network signal
            new_signal = network.next_time_instant()

            # run an online update
            regime_detection.read_new_time(np.copy(new_signal))

        # save the likelihood history
        aprob_history = regime_detection.aprob_history
        rho_estimate = regime_detection.rho_history

        # save the data in the structure to write to pickle
        for i in range(0, network_size):

            final_aprob = []

            for j in range(0, 2**(network_size-1)):

                final_aprob.append(aprob_history[i][j][-1])

                data_dict[i][j]["rho"].append(rho_estimate[i][j][-1])
                data_dict[i][j]["aprob"].append(aprob_history[i][j][-1])

            max_model_index = final_aprob.index(max(final_aprob))

            aprob_best_model_hist[i].append(max_model_index)

    # =================================================
    #              SAVE THE DATA
    # =================================================

    # create the dict to save
    save_dict = {}

    # parameter section of the dict to save
    save_dict["parameters"] = {"network size": network_size,
                               "adjacency matrix": adjacency_matrix,
                               "total time instants": t,
                               "total number of MC simulations": total_rep,
                               "Gibbs sampling iterations": gibbs_rep,
                               "rho true value": rho
                               }


    save_dict["data"] = data_dict

    # the file name
    file_name = "exp11-data-" + time_string + ".pickle"

    print(file_name)

    # save the file
    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == "__main__":

    main()

