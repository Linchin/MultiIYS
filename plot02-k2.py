__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      plot02-k2.py

#   DESCRIPTION:    Interactive Yule-Simon process

#                   s_A:    1 0 0 0 ... 0 0 1 0 0 ... 0 0
#                   s_B:    1 0 0 0 ... 0 0 0 0 0 ... 0 1
#                           |<---  k_1 ---->|<-- k_2 -->|

#                   M_0: A does NOT make B reset
#                   M_1: A makes B reset

#   PLOT:           p(M_0|k_2, alpha) vs. p(M_1|k_2, alpha)


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/27/2018 - 09/27/2018

#   UPDATE:         10/25/2018
#                   adapted for the ICASSP format
#                   got a really nice format and line style and so on.

# ------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import wishart, multivariate_normal, bernoulli, invwishart, beta
from scipy.stats import gamma as sci_gamma
from math import *
import matplotlib.pyplot as plt

def p_M0(k1, k2, alpha):

    p = 1

    i = 1

    while i <= k2:

        p *= (k1+i)/(k1+i+alpha)

        i += 1

    p *= alpha/(k1+k2+alpha)

    return p


def p_M1(k2, alpha):

    p = 1

    i = 1

    while i <= k2:
        p *= i / (i + alpha)

        i += 1

    p *= alpha / (k2 + alpha)

    return p



k2 = 50


line_1_M0 = list([p_M0(20,k,0.25) for k in range(0,k2+1)])
line_1_M1 = list([p_M1(k,0.25) for k in range(0,k2+1)])


line_1_M0 = np.array(line_1_M0)
line_1_M1 = np.array(line_1_M1)

temp = line_1_M0 + line_1_M1

line_1_M0 = line_1_M0 / temp
line_1_M1 = line_1_M1 / temp




line_2_M0 = list([p_M0(20,k,0.5) for k in range(0,k2+1)])
line_2_M1 = list([p_M1(k,0.5) for k in range(0,k2+1)])


line_2_M0 = np.array(line_2_M0)
line_2_M1 = np.array(line_2_M1)

temp = line_2_M0 + line_2_M1

line_2_M0 = line_2_M0 / temp
line_2_M1 = line_2_M1 / temp




line_3_M0 = list([p_M0(20,k,0.75) for k in range(0,k2+1)])
line_3_M1 = list([p_M1(k,0.75) for k in range(0,k2+1)])


line_3_M0 = np.array(line_3_M0)
line_3_M1 = np.array(line_3_M1)

temp = line_3_M0 + line_3_M1

line_3_M0 = line_3_M0 / temp
line_3_M1 = line_3_M1 / temp



line_4_M0 = list([p_M0(10,k,0.5) for k in range(0,k2+1)])
line_4_M1 = list([p_M1(k,0.5) for k in range(0,k2+1)])


line_4_M0 = np.array(line_4_M0)
line_4_M1 = np.array(line_4_M1)

temp = line_4_M0 + line_4_M1

line_4_M0 = line_4_M0 / temp
line_4_M1 = line_4_M1 / temp




line_5_M0 = list([p_M0(20,k,0.5) for k in range(0,k2+1)])
line_5_M1 = list([p_M1(k,0.5) for k in range(0,k2+1)])


line_5_M0 = np.array(line_5_M0)
line_5_M1 = np.array(line_5_M1)

temp = line_5_M0 + line_5_M1

line_5_M0 = line_5_M0 / temp
line_5_M1 = line_5_M1 / temp



line_6_M0 = list([p_M0(40,k,0.5) for k in range(0,k2+1)])
line_6_M1 = list([p_M1(k,0.5) for k in range(0,k2+1)])


line_6_M0 = np.array(line_6_M0)
line_6_M1 = np.array(line_6_M1)

temp = line_6_M0 + line_6_M1

line_6_M0 = line_6_M0 / temp
line_6_M1 = line_6_M1 / temp


#plt.figure(figsize=(7,4))
f, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True,figsize=(7,2.3))
plt.subplots_adjust(left=None, bottom=0.45, right=None, top=None,
                wspace=None, hspace=None)
linewidth = 1
markersize = 2.5
me = 3
fontsize = 7

ax1.plot(range(0,k2+1), line_1_M0, color="xkcd:blue", linestyle='-', linewidth = linewidth, markevery=me, label=r"$\mathcal{M}_0, \rho_B=0.25, k_1=20$")

ax1.plot(range(0,k2+1), line_2_M0, color="xkcd:red", linestyle='--', linewidth = linewidth, markevery=me, label=r"$\mathcal{M}_0, \rho_B=0.5,\  k_1=20$")

ax1.plot(range(0,k2+1), line_3_M0, color="xkcd:green", linestyle=':', linewidth = linewidth, markevery=me, label=r"$\mathcal{M}_0, \rho_B=0.75, k_1=20$")

ax1.plot(range(0,k2+1), line_1_M1, color="xkcd:blue", linestyle='-', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.25, k_1=20$")

ax1.plot(range(0,k2+1), line_2_M1, color="xkcd:red", linestyle='--', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.5,\  k_1=20$")

ax1.plot(range(0,k2+1), line_3_M1, color="xkcd:green", linestyle=':', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.75, k_1=20$")


ax1.set_ylim(bottom=-0.0,top=1.0)
ax1.set_xlim(left=0, right=50)
ax1.set_xlabel(r"$k_2$",fontsize=7)
ax1.set_ylabel(r"$p$",fontsize=7)
ax1.set_title("(a)", fontsize = fontsize)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), borderaxespad=0.5,frameon=False, fontsize=7, ncol=2)

ax1.tick_params(axis='both', direction='in', length = 2, width = 1)
ax2.tick_params(axis='both', direction='in', length = 2, width = 1)

fontdict = {'fontsize': 7}

xtick = [0, 10, 20, 30, 40, 50]
ax1.set_xticks(xtick)
ax1.set_xticklabels(xtick, fontdict=fontdict)
ax2.set_xticks(xtick)
ax2.set_xticklabels(xtick, fontdict=fontdict)

ytick = [0, 0.5, 1]
ax1.set_yticks(ytick)
ax1.set_yticklabels(ytick, fontdict=fontdict)
ax2.set_yticks(ytick)
ax2.set_yticklabels(ytick, fontdict=fontdict)


axis_width=0.6
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(axis_width)
  ax2.spines[axis].set_linewidth(axis_width)




ax2.plot(range(0,k2+1), line_4_M0, color="xkcd:blue", linestyle='-', linewidth = linewidth, label=r"$\mathcal{M}_0, \rho_B=0.5, k_1=10$")

ax2.plot(range(0,k2+1), line_5_M0, color="xkcd:red", linestyle='--', linewidth = linewidth, label=r"$\mathcal{M}_0, \rho_B=0.5, k_1=20$")

ax2.plot(range(0,k2+1), line_6_M0, color="xkcd:green", linestyle=':', linewidth = linewidth, label=r"$\mathcal{M}_0, \rho_B=0.5, k_1=40$")

ax2.plot(range(0,k2+1), line_4_M1, color="xkcd:blue", linestyle='-', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.5, k_1=10$")

ax2.plot(range(0,k2+1), line_5_M1, color="xkcd:red", linestyle='--', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.5, k_1=20$")

ax2.plot(range(0,k2+1), line_6_M1, color="xkcd:green", linestyle=':', linewidth = linewidth, markevery=me, marker = "|", markersize=markersize, label=r"$\mathcal{M}_1, \rho_B=0.5, k_1=40$")

ax2.set_xlabel(r"$k_2$", fontsize = fontsize)
#ax2.set_ylabel(r"$p$")
ax2.set_title("(b)", fontsize = fontsize)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), borderaxespad=0.5,frameon=False, fontsize=7, ncol=2)


plt.tight_layout()


plt.savefig("plot_k2.pdf",bbox_inches='tight')



















