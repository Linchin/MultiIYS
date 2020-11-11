# coding: utf-8

"""
File name:
exp32-iYS_real_data.py

09/08/2020
use the same channels but different time to double check the
result.
Use the 100-105s

09/07/2020
load the spike data and run through the model in
exp31-iYS_top_gibbs.py
and/or
exp28-iYS_top_gibbs.py

test with spike data
on channel 4, 21, 27

combine the program from exp31

update the IYSNetwork class so it reads from the data
directly

"""

import scipy.io as sio
import os
from os.path import dirname, join as pjoin
import numpy as np
import math


# load the spike data matrix

print(dirname)

dir_path = dirname(os.path.realpath(__file__))
print(dir_path)

mat_fname_c004 = dir_path + "\spike_data\M_20180530_merged-ch004pos1_spiketimes.mat"
mat_fname_c021 = dir_path + '\spike_data\M_20180530_merged-ch021pos1_spiketimes.mat'
mat_fname_c027 = dir_path + '\spike_data\M_20180530_merged-ch027pos1_spiketimes.mat'

print(mat_fname_c004)

mat_c004 = sio.loadmat(mat_fname_c004)['data']
mat_c021 = sio.loadmat(mat_fname_c021)['data']
mat_c027 = sio.loadmat(mat_fname_c027)['data']

print(mat_c004)
print(len(mat_c004))

c004_array = np.array(mat_c004).flatten()
c021_array = np.array(mat_c021).flatten()
c027_array = np.array(mat_c027).flatten()

# only use the first 5000 samples - 5 seconds at 1khz sampling rate
c004_5k = np.zeros(5000)
c021_5k = np.zeros(5000)
c027_5k = np.zeros(5000)

for item in c004_array:
    if item < 100:
        continue
    elif item >= 105:
        break
    else:
        c004_5k[math.floor((item-100)/0.001)] = 1

for item in c021_array:
    if item < 100:
        continue
    elif item >= 105:
        break
    else:
        c021_5k[math.floor((item-100)/0.001)] = 1

for item in c027_array:
    if item < 100:
        continue
    elif item >= 105:
        break
    else:
        c027_5k[math.floor((item-100)/0.001)] = 1

# save all data in an arrayay
all_channel_data = dict()
all_channel_data["c004_5k"] = c004_5k
all_channel_data["c021_5k"] = c021_5k
all_channel_data["c027_5k"] = c027_5k

print("aloha~!!")

# combine the stuff from exp31
import time
import pickle

from functions.F15_IYSNetwork_import_data import IYSNetwork_data
from functions.F14_IYSDetection_iYS_top_gibbs import IYSDetection_iYS_top_gibbs


def iYS_detection():
    """
    Main function.
    """
    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 3

    total_rep = 1       # repeat the entire process
    gibbs_rep = 50
    gibbs_alpha_rep = 30000
    rho = 0.75
    required_time = 1000
    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # data section of the dict to be saved
    data_dict = {}

    for rep_exp_index in range(0, total_rep):
        print("Current repetition: rep=", rep_exp_index+1,
              "/", total_rep)

        # =================================================
        #                      MODEL
        # =================================================

        # generate network topology
        # (directed network)
        # item[i][j]=1 means node i influences node j, 0 otherwise
        # item[i][i]=0 all the time though each node technically
        # influences themselves
        #adjacency_matrix = np.zeros((network_size, network_size))
        #adjacency_matrix[0, 1] = 1
        #adjacency_matrix[0, 2] = 1
        #adjacency_matrix[1, 0] = 1
        #adjacency_matrix[1, 2] = 1

        # create the i-YS network object instance
        network = IYSNetwork_data(all_channel_data, rho=rho)

        # create the i-YS detection object instance
        regime_detection = IYSDetection_iYS_top_gibbs(network_size,
                                              gibbs_rep,
                                              gibbs_alpha_rep,
                                              required_time)

        # evolve the network and detection objects
        # before the required regimes are reached
        for time_instant in range(required_time):

            # generate the network signal for the next time instant
            new_signal = network.next_time_instant()

            # run model selection online update
            regime_detection.read_new_time(np.copy(new_signal))

        # save the model selection results
        # each experiment is saved separately
#        aprob_history = regime_detection.aprob_history
        rho_history = regime_detection.rho_history
        signal_history = regime_detection.signal_history

        data_dict[rep_exp_index] = {}
 #       data_dict[rep_exp_index]["aprob"] = aprob_history
        data_dict[rep_exp_index]["rho"] = rho_history
        data_dict[rep_exp_index]["signals"] = signal_history

    # =================================================
    #              SAVE THE DATA
    # =================================================
    # create the dict to save
    # parameter section and data section
    save_dict = {"parameters": {"network size": network_size,
                                "required regimes": required_time,
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
    file_name = "exp32-data-" + time_string + "(gibbs_real_data).pickle"
    complete_file_name = pjoin(script_dir, rel_path_temp, file_name)
    print("Saved file name: ", file_name)

    # save the file
    with open(complete_file_name, 'wb') as handle:
        pickle.dump(save_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved successfully!")

    return 0

iYS_detection()
