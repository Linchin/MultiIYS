# coding: utf-8


"""
File Name:
plot10-exp36_plot_convergence.py

Based on:
exp36
plot08

Date:
10/26/2020

Plot the correct rate vs. time slots
The results are not directly read from the files,
but from the notes I took.

This plot generates the results of the topology
0 1 0
0 0 1
1 0 0

at time slots 200, 400, 600, 800 and 1000

Total reps: 25

"""


import numpy as np
import time
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt



# absolute dir the script is in
script_dir = os.path.dirname(__file__)

# file is saved in the following folder
rel_path_temp = "result_temp"

time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())
print("Time string: ", time_string)

# generate the name of the graph to be created
graph_name = "plot10-exp36_plot_convergence-" + time_string + ".pdf"


complete_file_name = join(script_dir, rel_path_temp, graph_name)

# input the data
correct_count_01 = [24, 25, 25, 25, 25]
correct_count_02 = [18, 24, 25, 25, 25]
correct_count_10 = [22, 23, 23, 24, 25]
correct_count_12 = [25, 25, 25, 25, 25]
correct_count_20 = [25, 25, 25, 25, 25]
correct_count_21 = [19, 22, 25, 25, 25]

x_label = [200, 400, 600, 800, 1000]


f, ax1 = plt.subplots(nrows=1, ncols=1,
                    figsize=(6.78, 4.6))

ax1.plot(x_label, correct_count_01, 'b-o', label="0→1")
ax1.plot(x_label, correct_count_02, 'r-x', label="0→2")
ax1.plot(x_label, correct_count_10, 'g-*', label="1→0")
ax1.plot(x_label, correct_count_12, 'y-^', label="1→2")
ax1.plot(x_label, correct_count_20, 'c->', label="2→0")
ax1.plot(x_label, correct_count_21, 'm-s', label="2→1")

ax1.legend()
ax1.set_title("Correct Gibbs sampling results")

ax1.set_xlabel("time slots")
ax1.set_ylabel("correct #/25")

ax1.set_ylim(bottom=0, top=27)

plt.savefig(complete_file_name)