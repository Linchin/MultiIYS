__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      histogram_of_results_1.py

#   DESCRIPTION:    reading the results produced by YSP_inter_bi_macro.py
#                   and draw a histogram with it.
#


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/18/2018 - 09/18/2018

# ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


alpha0 = np.array([])
alpha1 = np.array([])
M0M1 = np.array([])



with open("M1_10_test_result_09172018.txt") as textFile:

    for line in textFile:

        alpha0_temp, alpha1_temp, l0_temp, l1_temp = line.split()

        alpha0_temp = float(alpha0_temp)
        alpha1_temp = float(alpha1_temp)
        l0_temp = float(l0_temp)
        l1_temp = float(l1_temp)

        print([alpha0_temp, alpha1_temp, l0_temp, l1_temp])

        alpha0 = np.append(alpha0, alpha0_temp)
        alpha1 = np.append(alpha1, alpha1_temp)

        if l1_temp > l0_temp:
            M_temp = 1
        else:
            M_temp = 0


        M0M1 = np.append(M0M1, M_temp)


print(M0M1)


# plot with various axes scales
plt.figure(1)

# linear
plt.subplot(311)
plt.hist(alpha0)

plt.title('alpha0')
plt.grid(True)


# log
plt.subplot(312)
plt.hist(alpha1)

plt.title('alpha1')
plt.grid(True)


# symmetric log
plt.subplot(313)
plt.hist(M0M1)

plt.title('M0M1')
plt.grid(True)

plt.show()













