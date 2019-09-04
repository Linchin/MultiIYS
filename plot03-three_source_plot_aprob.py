# coding: utf-8

"""
Date: 08/01/2019
Author: Lingqing Gan @ Stony Brook University

Title: plot03-three_source_plot_aprob.py

Description: read the pickle file generated by exp07.


08/02/2019
this plot only shows the aprob result as a histogram.

Not the online version.

"""

import numpy as np
import time
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm

# input the file name
file_name = "exp07-data-20190903-142600" + ".pickle"

# read the file
with open (file_name, 'rb') as handle:
    data_save = pickle.load(handle)

# retrieve the parameters
parameter_dict = data_save["parameters"]

network_size = parameter_dict["network size"]
adjacency_matrix = parameter_dict["adjacency matrix"]
total_time = parameter_dict["total time instants"]
MC_rep = parameter_dict["total number of MC simulations"]
Gibbs_rep = parameter_dict["Gibbs sampling iterations"]
rho_true = parameter_dict["rho true value"]

for item in iter(parameter_dict):
    print(item, ":", parameter_dict[item])

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
color_vector = np.linspace(0, 1, 2 ** (network_size - 1))

a = color_vector[1]

bins = np.linspace(0, 1, 20)

i = 0

for ax in axs.reshape(-1):

    # Retrieve data for the current subplot
    current_node_history_alpha = data_save["data"][i]

    all_data = []
    label_list = []
    color_vector_plot = []

    for j in range(0, 2 ** (network_size - 1)):

        all_data.append(current_node_history_alpha[j]["aprob"])
        label_list.append(j)
        color_vector_plot.append(color_jet(color_vector[j]))

    ax.hist(all_data,
            label=label_list,
            bins=bins,
            color=color_vector_plot
            )

    ax.legend(loc="upper right")

    i += 1

plt.show()





