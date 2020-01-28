"""
exp24

"""


import pickle
import os
from os.path import join
import matplotlib.pyplot as plt


# absolute dir the script is in
script_dir = os.path.dirname(__file__)
rel_path_temp = "result_temp"

# the file name
file_name = "exp23-data-20200128-104859(YSP_bivariate_new_model_update).pickle"
complete_file_name = join(script_dir, rel_path_temp, file_name)
print("Saved file name: ", file_name)

# save the file
with open(complete_file_name, 'rb') as handle:
    data_save = pickle.load(handle)
    print("Data loaded successfully!")

partitions_est = data_save["data"]["estimated_regimes"]
partitions_true = data_save["data"]["true_regimes"]

T = 100

N_vector = range(1, T+1)

fig, ax = plt.subplots(1, 1)
for key in partitions_est:
    ax.plot(N_vector, partitions_est[key], linewidth=0.2, color="darkgray")
ax.plot(N_vector, partitions_true, linewidth=1, color="r")
plt.show()



