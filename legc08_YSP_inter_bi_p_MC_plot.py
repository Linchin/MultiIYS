__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      legc08_YSP_inter_bi_p_MC_plot.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           10/15/2018 - 10/??/2018

# ------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t as student
from scipy.special import beta
from math import sqrt
from math import log

# import sys
# print(sys.path)

import numpy as np
from scipy.stats import wishart, multivariate_normal, bernoulli, invwishart, beta
from scipy.stats import gamma as sci_gamma
from math import *
import matplotlib.pyplot as plt



# read the files

prob = np.array([])

with open("YSP_inter_bi_p_MC_result_10152018_prob.txt") as textFile:

    for line in textFile:

        prob = np.append(prob, float(line))


alpha = np.array([])

with open("YSP_inter_bi_p_MC_result_10152018_alpha.txt") as textFile:
    for line in textFile:
        alpha = np.append(alpha, float(line))


beta_dist_alpha = np.array([])

with open("YSP_inter_bi_p_MC_result_10152018_beta_dist_alpha.txt") as textFile:
    for line in textFile:
        beta_dist_alpha = np.append(beta_dist_alpha, float(line))


beta_dist_beta = np.array([])

with open("YSP_inter_bi_p_MC_result_10152018_beta_dist_beta.txt") as textFile:
    for line in textFile:
        beta_dist_beta = np.append(beta_dist_beta, float(line))


# draw samples from each


sample_size = 1000

alpha_sample = np.zeros(sample_size)

p_sample = np.zeros(sample_size)


prob_normalized = prob / np.sum(prob)

item_samples = np.random.choice(len(prob), size=sample_size, replace=True, p=prob_normalized)


for i in range(0,sample_size):

    index = item_samples[i]

    alpha_sample[i] = alpha[index]

    alpha_temp = beta_dist_alpha[index]

    beta_temp = beta_dist_beta[index]

    p_sample[i] = np.random.beta(a=alpha_temp, b=beta_temp, size=1)

fontsize = 7

fontdict = {'fontsize': 7}

plt.figure(1)
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False,figsize=(3.39,2.3))


ax1.hist(alpha_sample, color="xkcd:blue")

ax1.set_xlim(left = 0, right = 2)

ax1.set_title(r"(a) $p(\rho)$", fontsize = fontsize)




ax2.hist(p_sample, color="xkcd:blue")

ax2.set_xlim(left = 0, right = 1)

ax2.set_title(r"(b) $p(\pi)$", fontsize = fontsize)


axis_width=0.6
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(axis_width)
  ax2.spines[axis].set_linewidth(axis_width)


ax1.tick_params(axis='both', direction='in', length = 2, width = 1)
ax2.tick_params(axis='both', direction='in', length = 2, width = 1)


xtick1 = [0, 1, 2]
xtick2 = [0, 0.5, 1]
ax1.set_xticks(xtick1)
ax1.set_xticklabels(xtick1, fontdict=fontdict)
ax2.set_xticks(xtick2)
ax2.set_xticklabels(xtick2, fontdict=fontdict)

ytick = [0, 100, 200, 300]
ax1.set_yticks(ytick)
ax1.set_yticklabels(ytick, fontdict=fontdict)
ax2.set_yticks(ytick)
ax2.set_yticklabels(ytick, fontdict=fontdict)


plt.savefig("alpha_p_hist.pdf",bbox_inches='tight')











