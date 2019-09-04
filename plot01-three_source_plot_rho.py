# coding: utf-8

"""
Date: 08/01/2019
Author: Lingqing Gan @ Stony Brook University

Title: plot01-three_source_plot_rho.py

Description: read the pickle file generated by exp07.

08/02/2019
this plot only shows the gibbs sampling result.

"""

import numpy as np
import time
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm

# input the file name
name_string = "exp07-data-20190903-142600"
file_name = name_string + ".pickle"

print("file name: ", file_name)

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

# write the data to a easy-to-read txt file
data_file_name = name_string + ".txt"

with open (data_file_name, 'w+') as  handle:
    for item in iter(parameter_dict):
        print(item, ":", parameter_dict[item])
        handle.write(str(item)+": "+str(parameter_dict[item])+"\n")

handle.close()


# plot Gibbs sampling results - histogram

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

i = 0

bins = np.linspace(0, 3, 20)

for ax in axs.reshape(-1):

    # Retrieve data for the current subplot
    current_node_history_alpha = data_save["data"][i]

    all_data = []
    label_list = []
    color_vector_plot = []

    for j in range(0, 2 ** (network_size - 1)):


        all_data.append(current_node_history_alpha[j]["rho"])
        label_list.append(j)
        color_vector_plot.append(color_jet(color_vector[j]))

    ax.hist(all_data,
                label=label_list,
                bins=bins,
                color=color_vector_plot
                )

    # x-axis
#    time_axis = [i for i in range(0, len(current_node_history[0]))]

#    ax.set_xlim(min(time_axis), max(time_axis))
#    ax.set_ylim(-0.03, 1.03)

#    node_history_list = sorted(current_node_history.items())

    # cycle through all data line
 #   for item, c in zip(node_history_list, color_vector):
  #      l = item[0]
  #      j = item[1]

   #     current_color = color_jet(c)

    #    ax.plot(time_axis, j, color=current_color, label=l)

    ax.legend(loc="upper right")

    i += 1

pic_name = name_string + ".pdf"

plt.show()

# save the plot as a pdf file
#plt.savefig(pic_name)



# plot the a posterior prob according to models - histogram



