# coding: utf-8


"""
File name:
exp40-real_data_auto_run.py

11/10/2020
run the real data automatically.
based on exp38
use F17 and F15

"""

import scipy.io as sio
import os
from os.path import dirname, join
import numpy as np
import math
import statistics
import pdb
import traceback
import sys
import time
import matplotlib.pyplot as plt
import pickle

# parallel
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()

from functions.F15_IYSNetwork_import_data import IYSNetwork_data
from functions.F17_IYSDetection_iYS_top_gibbs_v3 import IYSDetection_iYS_top_gibbs

# directory
print(dirname)
dir_path = dirname(os.path.realpath(__file__))
print(dir_path)

# load single mat file of spike time
def load_seq(begin, end, channel):

    file_name = dir_path + "/spike_data/M_20180530_merged-" + channel + "_spiketimes.mat"
    this_channel = sio.loadmat(file_name)['data']
    data_array =  np.array(this_channel).flatten()
    total_length = math.floor((end-begin)*1000)+1

    spike_discrete = np.zeros(total_length)

    for item in data_array:
        if item < begin:
            continue
        elif item > end:
            break
        else:
            spike_discrete[math.floor((item - begin) / 0.001)] = 1

    print(sum(spike_discrete))
    return spike_discrete


# a function to calculate R
def r_hat(sequences):
    # sequences:    nested Lists
    # calculate the r_hat value for the Gibbs sequence convergence

    m = len(sequences)
    n = len(sequences[0])
    W_seq = [statistics.variance(_) for _ in sequences]
    W = statistics.mean(W_seq)

    mean_seq = [statistics.mean(_) for _ in sequences]
    mean_all = statistics.mean(mean_seq)

    temp = [(_-mean_all)**2 for _ in mean_seq]
    B = n/(m-1)*temp

    var_hat = (n-1)/n*W + 1/n*B
    r_hat = math.sqrt(var_hat/W)

    return r_hat


def iYS_gibbs(all_channel_data, required_time):
    """
    return the gibbs sample results
    """
    # =================================================
    #                    PARAMETERS
    # =================================================
    network_size = 2

    total_rep = 1       # repeat the entire process
    gibbs_rep = 400
    gibbs_alpha_rep = 30000
    rho = 0.75

    # data section of the dict to be saved
    data_dict = {}

    for rep_exp_index in range(0, total_rep):
        print("Current repetition: rep=", rep_exp_index+1,
              "/", total_rep)

        # =================================================
        #                      MODEL
        # =================================================

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
        model_count = regime_detection.model_count

        data_dict[rep_exp_index] = {}
 #       data_dict[rep_exp_index]["aprob"] = aprob_history
        data_dict[rep_exp_index]["rho"] = rho_history
        data_dict[rep_exp_index]["signals"] = signal_history

    # =================================================
    #              SAVE THE DATA
    # =================================================
    # create the dict to save
    # parameter section and data section
    # save_dict = {"parameters": {"network size": network_size,
    #                             "required regimes": required_time,
    #                             "total number of MC simulations": total_rep,
    #                             "Gibbs sampling iterations": gibbs_rep,
    #                             "rho true value": rho,
    #                             "data format": "rep, i, j"
    #                             },
    #              "data": data_dict}
    #
    # # absolute dir the script is in
    # script_dir = os.path.dirname(__file__)
    # rel_path_temp = "result_temp"
    #
    # # the file name
    # file_name = "exp40-data-" + time_string + "(real_data_auto_run).pickle"
    # complete_file_name = pjoin(script_dir, rel_path_temp, file_name)
    # print("Saved file name: ", file_name)
    #
    # # save the file
    # with open(complete_file_name, 'wb') as handle:
    #     pickle.dump(save_dict, handle,
    #                 protocol=pickle.HIGHEST_PROTOCOL)
    #     print("Data saved successfully!")

    return model_count

# parallel using class __call__()
class Gibbs_parallel_c():
    def __call__(self, item):
        channel_data_temp = channel_samples(channels, item[0], item[1])
        required_length = math.floor((item[1] - item[0]) * 1000) + 1
        results_temp = iYS_gibbs(channel_data_temp, required_length)
        total_temp = sum(results_temp)
        return results_temp / total_temp



def channel_samples(channels, begin_time, end_time):
    # return the dict of all sampled signals
    all_channel_data = dict()

    for item in channels:
        all_channel_data[item] = load_seq(begin_time, end_time, item)
    return all_channel_data

channels = ["ch021pos1", "ch099pos1"]

def test_all_delays(channels):
    # main

    # record running time
    start_time = time.time()

    # read the time stamps, or load from the pre-read data
    try:
        time_stamps_list_file_name = dir_path + "/shape_trials_time_stamps.pickle"
        with open(time_stamps_list_file_name, "rb") as input_file:
            time_stamps_list = pickle.load(input_file)
            print("Loaded time stamps.")
    except:
        file_name = dir_path + "/spike_data/M_20180530_merge_events.mat"
        loaded_events = sio.loadmat(file_name)
        # Abs_Cue_S
        # Conc_Cue_Rec
        # Conc_Cue_Oval
        # Conc_Cue_Bow
        Abs_Cue_S = loaded_events['Abs_Cue_S'].flatten()
        Con_Cue_Rec = loaded_events['Conc_Cue_Rec'].flatten()
        Con_Cue_Oval = loaded_events['Conc_Cue_Oval'].flatten()
        Con_Cue_Bow = loaded_events['Conc_Cue_Bow'].flatten()
        Con_Cue_combined = np.concatenate((Con_Cue_Rec,
                                           Con_Cue_Oval,
                                           Con_Cue_Bow))
        Con_Cue_combined_sorted = np.sort(Con_Cue_combined)

        shape_delay1_time = []
        skip_count = 0
        for start in Abs_Cue_S:
            start_time = start + 0.45
            # print(start_time)
            for item in Con_Cue_combined_sorted:
                if item < start_time:
                    continue
                elif item > start_time:
                    if item - start_time < 0.5:
                        continue
                    elif 0.5 <= item - start_time <= 0.9:
                        shape_delay1_time.append((start_time, item))
                        break
                    else:
                        skip_count += 1
                        print("Delay1 longer than 900ms, skipped this interval.")
                        print("Abs_Cue_S time:", start)
                        print("delay1 start time:", start_time)
                        print("Conc_Cue time:", item)
                        print("Delay1 length:", item - start_time)
                        print("Total skipped so far:", skip_count)
                        print("\n")
                        break
                        # print("Error in sorting data.")
                        # exit(-1)

        with open(dir_path + "/shape_trials_time_stamps.pickle", 'wb') as handle:
            pickle.dump(shape_delay1_time, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved time stamps.")
        time_stamps_list = shape_delay1_time

    # shorten the time_stamps_list just to make debug faster
    #temp = time_stamps_list[:100]
    #time_stamps_list = temp.copy()

    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    print("Time string: ", time_string)

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    rel_path_plot = "plots"

    results = []

    time_slots_count = 0

    # parallel
    # def gibbs_parallel(item):
    #     channel_data_temp = channel_samples(channels, item[0], item[1])
    #     required_length = math.floor((item[1]-item[0])*1000)+1
    #     results_temp = iYS_gibbs(channel_data_temp, required_length)
    #     total_temp = sum(results_temp)
    #     return results_temp/total_temp

    gibbs_p_obj = Gibbs_parallel_c()
    results = Parallel(n_jobs=num_cores-6)(delayed(gibbs_p_obj)(item)for item in time_stamps_list)

    # none parallel version
    # for each delay
    # for item in time_stamps_list:
    #     time_slots_count += 1
    #     print("Index of time slots:", time_slots_count, "/", len(time_stamps_list))
    #     channel_data_temp = channel_samples(channels, item[0], item[1])
    #     required_length = math.floor((item[1]-item[0])*1000)
    #     results_temp = iYS_gibbs(channel_data_temp, required_length)
    #     total_temp = sum(results_temp)
    #     results.append(results_temp/total_temp)

    # plot the hist
    results_T = np.array(results).T.tolist()
    f, axes = plt.subplots(nrows=2, ncols=2,
                        figsize=(6.78, 4.6))

    # flatten axes
    flat_axes = [item for sublist in axes for item in sublist]

    bins = np.linspace(0, 1, num=20)

    for i in range(len(results_T)):
        flat_axes[i].hist(results_T[i], bins=bins)
        flat_axes[i].set_title(str(i))
        flat_axes[i].set_xlim(left=0, right=1.05)

    # file name
    file_name = "exp40-plot_iYS_hist-" + time_string + ".pdf"
    complete_file_name = join(script_dir, rel_path_plot,
                              file_name)


    #plt.show()
    plt.savefig(complete_file_name, bbox_inches='tight')

    print("My program took", round((time.time() - start_time)/60, 2), "min to run.")
    print("0")
    return 0


if __name__ == '__main__':
    try:
        test_all_delays(channels)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


