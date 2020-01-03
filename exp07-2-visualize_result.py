"""
exp07-2-visualize_result.py

12/30/2019

read the pickle files generated by exp07 and plot the histogram of the
results.
"""

import os
from os.path import join
import pickle
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np

# folder to save the model selection results
rel_path_detection = "data05-topology001-100-000"

script_dir = os.path.dirname(__file__)

network_size = 3
time = 100

# the file name
file_name = "exp07-data-20190807-105657.pickle"
print("Saved file name: ", file_name)

# read the dict from the pickle file
with open(join(script_dir, rel_path_detection, file_name), 'rb') as handle:
    data_dict = pickle.load(handle)
    print("Model selection results loaded successfully!")


model_selected = {}
# obtain the model selection results and
for i in range(0, network_size):
    model_selected[i] = []
    for t in range(0, time):
        aprob = [data_dict["data"][i][j]["aprob"][t] for j in range(0, 2**(network_size-1))]
        model_selected[i].append(aprob.index(max(aprob)))

# count
count = {}
for i in range(0, network_size):
    count[i] = []
    for j in range(0, 2**(network_size-1)):
        count[i].append(model_selected[i].count(j))


print(count)

fig, axes = plt.subplots(network_size,1, figsize=(4,6))
bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
node_new_order = [2, 0, 1]
hfont = {'fontname':'Trebuchet MS'}

for i in range(network_size):

    axes[i].hist(model_selected[int(node_new_order[i])], bins=bins, color="firebrick", rwidth=0.6)
#    for item in range(len(bins)-1):
#        plt.text(arr[1][item], arr[0][item], str(arr[0][item]))
    axes[i].set_xticks([0, 1, 2, 3])
    axes[i].set_xticklabels([r"$H_0$", r"$H_1$", r"$H_2$", r"$H_3$"], **hfont)
    axes[i].set_yticks([0, 25, 50, 75, 100])
    axes[i].set_yticklabels(["0", "25", "50", "75", "100"], **hfont)
    axes[i].set_ylim(bottom=0, top=100)
    axes[i].set_title("Node "+str(i+1), **hfont)

fig_path = "data05-topology001-100-000"
plt.tight_layout()
plt.show()
#plt.savefig(join(script_dir, fig_path, "model_selection_result-data05-topology001-100-000.pdf"))


print("1")


