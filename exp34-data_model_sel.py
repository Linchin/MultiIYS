# coding: utf-8

"""
File names:
exp34-data_model_sel.py

Lingqing Gan

09/22/2020
This model uses an ABC like method to decide which one of the four
models the real data is generated from.

We only use a two node network here.
We run the generative model under each of the four models, and a
range of the rho values in each of the four models. We have done
this step and saved the results in exp33.

We compare the frequency of 1s in the real data and the generative
data, for each of the four hypotheses individually. Thus we decide
the rho value for each of the models. Then based on the rho value,
we calculate the likelihood given each of the four hypotheses. Then
we select the hypothesis with the highest likelihood.

"""

import pickle
import numpy as np

import time
import os
from os.path import join, dirname
import scipy.io as sio
import math


time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())

# dictionary that saves the data (now use the mean value of the data)
# sigs = {"00": {0.1:[[ave_sig0], [ave_sig1]], 0.2:[], ..., 2.0[]},
#         "01": {...},
#         "10": {...},
#         "11": {...}}
# where "01" means node 0 affects node 1 but not vice versa.

# load the saved generative data

# absolute dir the script is in
script_dir = os.path.dirname(__file__)
rel_path_temp = "result_temp"

# file time string
file_time_string = "20200922-192608"

# the file name
file_name = "exp33-data-" + file_time_string + "(sigfreq_vs_rho).pickle"
complete_file_name = join(script_dir, rel_path_temp, file_name)

# load the file
with open(complete_file_name, 'rb') as handle:
    sigs = pickle.load(handle)
    print("Generative data loaded successfully!")

# load the experiment data
data_path = "spike_data"

dir_path = dirname(os.path.realpath(__file__))

mat_fname_c004 = dir_path + "\spike_data\M_20180530_merged-ch004pos1_spiketimes.mat"
mat_fname_c021 = dir_path + '\spike_data\M_20180530_merged-ch021pos1_spiketimes.mat'
mat_fname_c027 = dir_path + '\spike_data\M_20180530_merged-ch027pos1_spiketimes.mat'
mat_fname_c065 = dir_path + '\spike_data\M_20180530_merged-ch065pos1_spiketimes.mat'
mat_fname_c083 = dir_path + '\spike_data\M_20180530_merged-ch083pos1_spiketimes.mat'


print(mat_fname_c004)

mat_c004 = sio.loadmat(mat_fname_c004)['data']
mat_c021 = sio.loadmat(mat_fname_c021)['data']
mat_c027 = sio.loadmat(mat_fname_c027)['data']
mat_c065 = sio.loadmat(mat_fname_c065)['data']
mat_c083 = sio.loadmat(mat_fname_c083)['data']

c004_array = np.array(mat_c004).flatten()
c021_array = np.array(mat_c021).flatten()
c027_array = np.array(mat_c027).flatten()
c065_array = np.array(mat_c065).flatten()
c083_array = np.array(mat_c083).flatten()


# only use the first 5000 samples - 5 seconds at 1khz sampling rate
c004_5k = np.zeros(5000)
c021_5k = np.zeros(5000)
c027_5k = np.zeros(5000)
c065_5k = np.zeros(5000)
c083_5k = np.zeros(5000)

for item in c004_array:
    if item < 0:
        continue
    elif item >= 5:
        break
    else:
        c004_5k[math.floor((item)/0.001)] = 1

for item in c021_array:
    if item < 0:
        continue
    elif item >= 5:
        break
    else:
        c021_5k[math.floor((item)/0.001)] = 1

for item in c027_array:
    if item < 0:
        continue
    elif item >= 5:
        break
    else:
        c027_5k[math.floor((item)/0.001)] = 1

for item in c065_array:
    if item < 0:
        continue
    elif item >= 5:
        break
    else:
        c065_5k[math.floor((item)/0.001)] = 1

for item in c083_array:
    if item < 0:
        continue
    elif item >= 5:
        break
    else:
        c083_5k[math.floor((item)/0.001)] = 1


# save all data in an arrayay
all_channel_data = dict()
all_channel_data["c004_5k"] = c004_5k
all_channel_data["c021_5k"] = c021_5k
all_channel_data["c027_5k"] = c027_5k
all_channel_data["c065_5k"] = c065_5k
all_channel_data["c083_5k"] = c083_5k

print("aloha~!! Loaded all experiment data.")

# compare likelihood






