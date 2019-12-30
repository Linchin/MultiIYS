# coding: utf-8

"""
File name:
exp07-three_source_multi.py

Author:
Lingqing Gan @ Stony Brook University

07/08/2019 notes
Description:
1. Run the three node models multiple times;
2. Collect data according to the given format;
3. Save the data into a pickle data file.
4. The data file will be read and plotted by exp09.py.

09/06/2019 notes
added the code to save the original signals.
Then we can analyze it.

12/30/2019 notes
Compared with the file exp07-v2, this should be the later
version. We should stick with this one.
Changed the network class from F07_stable_01 into F07_stable_02

"""

import numpy as np
import time
import pickle

from functions.F07_IYSNetwork_stable_02 import IYSNetwork
from functions.F08_IYSDetection_stable_01 import IYSDetection


def main():
    """
    Main function.
    """

    # =================================================
    #                    PARAMETERS
    # =================================================

    network_size = 3
    t = 1000           # total number of time instants
    total_rep = 100
    gibbs_rep = 10000
    rho = 0.75

    time_string = time.strftime("%Y%m%d-%H%M%S",
                                time.localtime())

    aprob_best_model_hist = {}

    # data section of the dict to save
    data_dict = {}

    for i in range(0, network_size):
        aprob_best_model_hist[i] = []
        data_dict[i] = {"signal":[]}
        for j in range(0, 2**(network_size-1)):
            data_dict[i][j] = {"rho": [],
                               "aprob": []}

    for rep_exp_index in range(0, total_rep):
        print("Current repetition: rep =", rep_exp_index)

        # =================================================
        #                      MODEL
        # =================================================

        # generate network topology
        # directed network.
        # item[i][j]=1 means node i influences node j
        # item[i][i]=0 all the time though each node technically
        # influences themselves
        adjacency_matrix = np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]])

        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix, rho=rho)

        # create the i-YS detection object instance
        regime_detection = IYSDetection(network_size, gibbs_rep)

        # for each time instant:
        for i in range(0, t):

            # generate the network signal
            new_signal = network.next_time_instant()

            # run an online update
            regime_detection.read_new_time(np.copy(new_signal))

            # save the signal history to data_dict
            for j in range(0, network_size):
                data_dict[j]["signal"].append(new_signal[j])

        # save the likelihood history
        aprob_history = regime_detection.aprob_history
        rho_estimate = regime_detection.rho_history

        # obtain the model selection results and
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
    file_name = "exp07-data-" + time_string + ".pickle"

    print(file_name)

    # save the file
    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == "__main__":

    main()

