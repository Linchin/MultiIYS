# coding: utf-8


"""
File name:
exp33-sigfreq_vs_rho.py

Lingqing Gan

09/22/2020
save and plot the average freq of 1s given rho value and topology.




09/21/2020

With a two node model, we generate the random signal under
varying rho and hypothesis. We save the signal data, esp.
the frequency of 1s. This data is then compared with the
real data, thus decide the values of the parameters in the
real data.

"""

from functions.F07_IYSNetwork_stable_02 import IYSNetwork
import numpy as np
import pickle
import time
import os
from os.path import join
import matplotlib.pyplot as plt

time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())



# dictionary that saves the data (now use the mean value of the data)
# sigs = {"00": {0.1:[[ave_sig0], [ave_sig1]], 0.2:[], ..., 2.0[]},
#         "01": {...},
#         "10": {...},
#         "11": {...}}
# where "01" means node 0 affects node 1 but not vice versa.

sigs = {"00": {},
        "01": {},
        "10": {},
        "11": {}}

rho_data_points = 100

sig_ave00_0 = np.zeros(rho_data_points)
sig_ave01_0 = np.zeros(rho_data_points)
sig_ave10_0 = np.zeros(rho_data_points)
sig_ave11_0 = np.zeros(rho_data_points)

sig_ave00_1 = np.zeros(rho_data_points)
sig_ave01_1 = np.zeros(rho_data_points)
sig_ave10_1 = np.zeros(rho_data_points)
sig_ave11_1 = np.zeros(rho_data_points)

sigs["00"][0] = sig_ave00_0
sigs["01"][0] = sig_ave01_0
sigs["10"][0] = sig_ave10_0
sigs["11"][0] = sig_ave11_0

sigs["00"][1] = sig_ave00_1
sigs["01"][1] = sig_ave01_1
sigs["10"][1] = sig_ave10_1
sigs["11"][1] = sig_ave11_1

total_time = 2000
total_rep = 100

aj_matrix00 = np.zeros((2, 2))
aj_matrix01 = np.zeros((2, 2))
aj_matrix10 = np.zeros((2, 2))
aj_matrix11 = np.zeros((2, 2))

aj_matrix01[0][1] = 1
aj_matrix10[1][0] = 1

aj_matrix11[0][1] = 1
aj_matrix11[1][0] = 1

rho = 0
for i in range(rho_data_points):
    rho += 0.1
    print("rho value: ", rho)

    for rep in range(total_rep):

        network00 = IYSNetwork(aj_matrix00, rho)
        network01 = IYSNetwork(aj_matrix01, rho)
        network10 = IYSNetwork(aj_matrix10, rho)
        network11 = IYSNetwork(aj_matrix11, rho)

        for t in range(total_time):
            network00.next_time_instant()
            network01.next_time_instant()
            network10.next_time_instant()
            network11.next_time_instant()

        sigs["00"][0][i] += np.average(network00.signal_history[0])/total_rep
        sigs["01"][0][i] += np.average(network01.signal_history[0])/total_rep
        sigs["10"][0][i] += np.average(network10.signal_history[0])/total_rep
        sigs["11"][0][i] += np.average(network11.signal_history[0])/total_rep

        sigs["00"][1][i] += np.average(network00.signal_history[1])/total_rep
        sigs["01"][1][i] += np.average(network01.signal_history[1])/total_rep
        sigs["10"][1][i] += np.average(network10.signal_history[1])/total_rep
        sigs["11"][1][i] += np.average(network11.signal_history[1])/total_rep


# save the data

# absolute dir the script is in
script_dir = os.path.dirname(__file__)
rel_path_temp = "result_temp"

# the file name
file_name = "exp33-data-" + time_string + "(sigfreq_vs_rho).pickle"
complete_file_name = join(script_dir, rel_path_temp, file_name)
print("Saved file name: ", file_name)

# save the file
with open(complete_file_name, 'wb') as handle:
    pickle.dump(sigs, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
    print("Data saved successfully!")

# plot the data
N_vector = range(1, rho_data_points+1)

fig1, ax1 = plt.subplots()


ax1.plot(N_vector, sig_ave00_0, label="sig_ave00_0")
ax1.plot(N_vector, sig_ave01_0, label="sig_ave01_0")
ax1.plot(N_vector, sig_ave10_0, label="sig_ave10_0")
ax1.plot(N_vector, sig_ave11_0, label="sig_ave11_0")

ax1.plot(N_vector, sig_ave00_1, label="sig_ave00_1")
ax1.plot(N_vector, sig_ave01_1, label="sig_ave01_1")
ax1.plot(N_vector, sig_ave10_1, label="sig_ave10_1")
ax1.plot(N_vector, sig_ave11_1, label="sig_ave11_1")


ax1.legend(fontsize=14)
file_name = "exp33-data-" + time_string + "-simu_sig_freq_plot.pdf"
complete_file_name = join(script_dir, rel_path_temp, file_name)
plt.savefig(complete_file_name)













