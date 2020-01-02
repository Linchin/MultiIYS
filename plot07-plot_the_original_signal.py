# coding: utf-8

"""
plot07-plot_the_original_signal.py

09062019

Lingqing Gan

Description:
We use this code to plot the signal pattern.

"""

import pickle

# input the file name
name_string = "exp07-data-20190906-134627"
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

with open (data_file_name, 'w+') as handle:
    for item in iter(parameter_dict):
        print(item, ":", parameter_dict[item])
        handle.write(str(item)+": "+str(parameter_dict[item])+"\n")

handle.close()

# write the original signal sequences
data_file_name_2 = name_string + "-signals" + ".txt"

with open(data_file_name_2, 'w+') as handle:
    for i in range(0, network_size):
        handle.write(str(i)+" ")
    handle.write("\n")

    for j in range(0, total_time):
        for i in range(0, network_size):
            handle.write(str(int(data_save["data"][i]["signal"][j]))+" ")
        handle.write("\n")

handle.close()
















