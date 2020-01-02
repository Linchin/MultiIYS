# coding: utf-8

"""
File Name:
exp17-parsed_testing_stable.py

Based on:
exp07-three_source_multi_parsed.py
exp07-three_source_multi.py
exp11-online_update.py
exp13-three_source_multi_parsed.py
exp14-three_source_multi_parsed_02.py
exp15-parsed_testing.py

Author: Lingqing Gan @ Stony Brook University

12/29/2019
Didn't do it here. We have another version that runs on real data
that's in the real data project. Please see that project.

12/23/2019 notes
This is the stable version of the code.
It is changed so we can run real data.


12/02/2019 note
development after exp14.
The code finally worked. Now we need further tests.
1. adjust the details so it works on multi-node scenarios (done)
2. see if it's correct

10/31/2019 note
we alter the first signal to be all 1s instead of all 0s.

09/23/2019 note
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

Past Description:
1. Run the three node models multiple times;
2. Collect data according to the given format;
3. Save the data into a pickle data file.
4. The data file will be read and plotted by exp09.py.

09/06/2019 notes
added the code to save the original signals.
Then we can analyze it.
"""

import numpy as np
import time
import pickle

from functions.F07_IYSNetwork_stable_02 import IYSNetwork
from functions.F11_IYSDetection_parse_03 import IYSDetection_parse


def main():
    """
    Main function.
    """
    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 3
    t = 1000           # total number of time instants
    total_rep = 50
    gibbs_rep = 20000
    rho = 0.75
    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # data section of the dict to be saved
    data_dict = {}

    for rep_exp_index in range(0, total_rep):
        print("Current repetition: rep=", rep_exp_index)
        # =================================================
        #                      MODEL
        # =================================================
        # generate network topology
        # (directed network)
        # item[i][j]=1 means node i influences node j, 0 otherwise
        # item[i][i]=0 all the time though each node technically
        # influences themselves
        adjacency_matrix = np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]])
        # adjacency_matrix = np.array([[0, 0],
        #                             [1, 0]])
        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix, rho=rho)
        # create the i-YS detection object instance
        regime_detection = IYSDetection_parse(network_size,
                                              gibbs_rep, t)
        # evolve the network and detection objects
        for i in range(0, t):
            # generate the network signal for the next time instant
            new_signal = network.next_time_instant()

            # save the new signals
            # for j in range(0, network_size):
            #    data_dict[j]["signal"].append(new_signal[j])

            # run model selection online update
            regime_detection.read_new_time(np.copy(new_signal))

        # save the model selection results
        # each experiment is saved separately
        aprob_history = regime_detection.aprob_history
        rho_history = regime_detection.rho_history
        signal_history = regime_detection.signal_history

        data_dict[rep_exp_index] = {}
        data_dict[rep_exp_index]["aprob"] = aprob_history
        data_dict[rep_exp_index]["rho"] = rho_history
        data_dict[rep_exp_index]["signals"] = signal_history

    # =================================================
    #              SAVE THE DATA
    # =================================================
    # create the dict to save
    # parameter section and data section
    save_dict = {"parameters": {"network size": network_size,
                                "adjacency matrix": adjacency_matrix,
                                "total time instants": t,
                                "total number of MC simulations": total_rep,
                                "Gibbs sampling iterations": gibbs_rep,
                                "rho true value": rho,
                                "data format": "rep, i, j"
                                },
                 "data": data_dict}
    # the file name
    file_name = "exp16-data-" + time_string + ".pickle"
    print("Saved file name: ", file_name)
    # save the file
    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved successfully!")
    return 0


if __name__ == "__main__":
    main()
