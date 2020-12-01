# coding: utf-8

"""
file name:
exp39-plot_iYS_hist.py

Plot the histogram of regime lengths and see what
they look like. Maybe fit it.
"""

import numpy as np
import time
import os
from os.path import join
import pdb
import traceback
import sys
import matplotlib.pyplot as plt
import math


from functions.F07_IYSNetwork_stable_02 import IYSNetwork


def main():
    """
    Main function.
    """

    time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    print("Time string: ", time_string)

    network_size = 2
    rho = 0.75
    required_time = 200000

    adjacency_matrix = np.zeros((network_size, network_size))
    adjacency_matrix[0, 1] = 1

    # create the i-YS network object instance
    network = IYSNetwork(adjacency_matrix, rho=rho)

    for time_instant in range(required_time):
        # generate the network signal for the next time instant
        network.next_time_instant()

    # read the signals from the network object
    read_signals = network.signal_history[1]

    regimes = []
    begin = -1
    for i in range(len(read_signals)):
        if read_signals[i] == 1:
            regimes.append(i-begin)
            begin = i
    print(max(regimes), min(regimes))

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    rel_path_plot = "plots"

    # file name
    file_name = "exp39-plot_iYS_hist-" + time_string +".pdf"
    complete_file_name = join(script_dir, rel_path_plot,
                              file_name)

    # plot and save the graph
    f, ax1 = plt.subplots(nrows=1, ncols=1,
                        figsize=(6.78, 4.6))

    w = 1
    n = math.ceil((max(regimes) - min(regimes)) / w)
    ax1 = plt.hist(regimes, bins=n, log=True)

    # ax1.hist(regimes)
    plt.savefig(complete_file_name, bbox_inches='tight')

    return 0



if __name__ == '__main__':
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
