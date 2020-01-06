# coding: utf-8

"""
File Name:
exp20-parsed_testing_determined_regime_corrected_iYS.py

Based on:
exp07-three_source_multi_parsed.py
exp07-three_source_multi.py
exp11-online_update.py
exp13-three_source_multi_parsed.py
exp14-three_source_multi_parsed_02.py
exp15-parsed_testing.py
exp16-parsed_testing_stable.py
F12_IYSDetection_parse_determ_regm.py
F13_IYSDetection_parse_determ_regm_corrected_iYS.py


Author: Lingqing Gan @ Stony Brook University

01/06/2020 notes (exp20)
A version that corrected the iYS prob function.

01/06/2020 notes (exp19)
not sure if this is gonna work, since the length of the
regimes might not be independent.
Also, I found that there was an error with the formula of
the iYS likelihood function.
So I will work on exp20 first, where I correct the iYS
likelihood formula.

01/04/2020 notes (exp19)
Because the results in exp18 still didn't work
with >3 node networks, and a lot of debugging still
didn't make it work. So now I'm trying out plan B.

The inspiration came from a discussion with Prof. this
past Thursday(01/02/2020). Professor mentioned that we
haven't accounted for the variation involved while we
estimate the parameter rho. This reminded me of something
that I thought might have influenced the accuracy of the
model selection results. When we are deciding whether the
neighboring nodes are influencing the node of interest,
we do a separate Gibbs sampling for each node-neighbor
pair. Might there be any overfitting?

So the idea here is, for each node of interest, we estimate
the parameter rho using only its regimes that has definitely
no influencer. Then we use this same estimated rho value
for all model selection pertaining to this node.


01/02/2019 notes (exp18)
For now it only works on two node network. For 3
node it doesn't work correctly.

12/29/2019 notes (exp18)
This version is based on the stable version exp16.
The improvement is to make the fixed total time
slots a varying variable, and we only control the
total number of (unambiguous) regimes each
node-neighbor pair uses. Once the number of regimes
reaches the threshold, we stop model selection with
this pair.

For this reason, we need to change the detection class
that we originally used from F11_IYSDetection_parse_03.py.
Instead, we use a new file, F12_IYSDetection_parse_determ_regm.py
for IYS model selection.

12/23/2019 notes (exp16)
Now we make a stable version of the code.
1. We run it under various scenarios to collect data;
2. We try running it on real data.

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
import os
from os.path import join


from functions.F07_IYSNetwork_stable_02 import IYSNetwork
from functions.F13_IYSDetection_parse_determ_regm_corrected_iYS import IYSDetection_parse_dtm_rgm_crtd_iYS



def main():
    """
    Main function.
    """
    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 3

    total_rep = 1
    gibbs_rep = 20000
    rho = 0.75
    regimes_required = 15
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
        adjacency_matrix = np.zeros((network_size, network_size))
        # adjacency_matrix[0, 1] = 1
        # adjacency_matrix[1, 0] = 1

        # create the i-YS network object instance
        network = IYSNetwork(adjacency_matrix, rho=rho)

        # create the i-YS detection object instance
        regime_detection = IYSDetection_parse_dtm_rgm_crtd_iYS(network_size,
                                              gibbs_rep, regimes_required)

        # evolve the network and detection objects
        # before the required regimes are reached
        while not regime_detection.regime_reached:

            # generate the network signal for the next time instant
            new_signal = network.next_time_instant()

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
                                "required regimes": regimes_required,
                                "total number of MC simulations": total_rep,
                                "Gibbs sampling iterations": gibbs_rep,
                                "rho true value": rho,
                                "data format": "rep, i, j"
                                },
                 "data": data_dict}

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    rel_path_temp = "result_temp"

    # the file name
    file_name = "exp20-data-" + time_string + "(determined_regime_corrected_iYS).pickle"
    complete_file_name = join(script_dir, rel_path_temp, file_name)
    print("Saved file name: ", file_name)
    # save the file
    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved successfully!")
    return 0


if __name__ == "__main__":
    main()
