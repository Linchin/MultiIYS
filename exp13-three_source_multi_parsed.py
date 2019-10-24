# coding: utf-8

"""
Date: 09/23/2019
Author: Lingqing Gan @ Stony Brook University

Title: exp07-three_source_multi_parsed.py
Based on: exp07-three_source_multi.py

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

Check:
1.
2.
3.
4.

Past Description:
1. Run the three node models multiple times;
2. Collect data according to the given format;
3. Save the data into a pickle data file.
4. The data file will be read and plotted by exp09.py.

20190906 notes
added the code to save the original signals.
Then we can analyze it.
"""

import numpy as np
import time
import pickle

from functions.F07_IYSNetwork_stable_01 import IYSNetwork
from functions.F10_IYSDetection_parse import IYSDetection_parse


def main():
    """
    Main function.
    """
    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 2
    t = 1000           # total number of time instants
    total_rep = 1
    gibbs_rep = 10000
    rho = 0.75

    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # data section of the dict to save
    data_dict = {}
    for i in range(0, network_size):
        data_dict[i] = {"signal":[]}
        for j in range(0, 2**(network_size-1)):
            data_dict[i][j] = {"rho": {"aln":[],
                                       "ifcd":[]},
                               "aprob": {"aln":[],
                                         "ifcd":[]}}

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
        # adjacency_matrix = np.array([[0, 0, 0],
        #                              [0, 0, 0],
        #                              [0, 0, 0]])
        adjacency_matrix = np.array([[0,0],
                                     [0,0]])
        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix, rho=rho)
        # create the i-YS detection object instance
        regime_detection = IYSDetection_parse(network_size,
                                              gibbs_rep, t)
        # Generate the signal
        for i in range(0, t):
            # generate the network signal
            new_signal = network.next_time_instant()
            # save the new signals
            for j in range(0, network_size):
                data_dict[j]["signal"].append(new_signal[j])
            # run model selection online update
            regime_detection.read_new_time(np.copy(new_signal))

        # save the model selection results
        aprob_history = regime_detection.aprob_history
        rho_history = regime_detection.rho_history
        for i in range(0, network_size):
            for j in range(0, network_size):
                print(rho_history[i][j], aprob_history[i][j])
                if i == j:
                    continue
                data_dict[i][j]["rho"]["aln"].append(rho_history[i][j][-1][0])
                data_dict[i][j]["rho"]["ifcd"].append(rho_history[i][j][-1][1])
                data_dict[i][j]["aprob"]["aln"].append(aprob_history[i][j][-1][0])
                data_dict[i][j]["aprob"]["ifcd"].append(aprob_history[i][j][-1][1])

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
    file_name = "exp13-data-" + time_string + ".pickle"
    print(file_name)
    # save the file
    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    main()