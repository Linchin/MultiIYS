

import pickle
from os.path import dirname, join
import os

# directory
print(dirname)
dir_path = dirname(os.path.realpath(__file__))
print(dir_path)



# read the results
gibbs_results_file_name = dir_path + "/result_temp/exp41-all_gibbs_20201122.pickle"
with open(gibbs_results_file_name, 'rb') as handle:
    results = pickle.load(handle)

for key in results:
    print(key)
    print(results[key])