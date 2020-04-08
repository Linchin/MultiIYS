__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      plot06-histogram_of_results_2_v2.py

#   DESCRIPTION:    reading the results produced by legc04_YSP_inter_bi_bidir_macro.py
#                   and draw histogram(s) with it.
#


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           11/04/2018 - 11/04/2018

#   UPDATE 01:      10/25/2018
#                   improved format for icassp

#   UPDATE 02:      changed into two histograms as per Prof's requirement.
#                   Looks more concise.

# ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt



M2_A = np.array([])
M2_B = np.array([])

M1_A = np.array([])
M1_B = np.array([])

M0_A = np.array([])
M0_B = np.array([])


with open("M2_AB_test_result_09262018.txt") as textFile:

    for line in textFile:

        l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B = line.split()

        l0_temp_A = float(l0_temp_A)
        l1_temp_A = float(l1_temp_A)

        l0_temp_B = float(l0_temp_B)
        l1_temp_B = float(l1_temp_B)

        print([l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B])


        if l1_temp_A > l0_temp_A:
            M_temp_A = 1
        else:
            M_temp_A = 0

        if l1_temp_B > l0_temp_B:
            M_temp_B = 1
        else:
            M_temp_B = 0


        M2_A = np.append(M2_A, M_temp_A)
        M2_B = np.append(M2_B, M_temp_B)


with open("M1_AB_test_result_09262018.txt") as textFile:
    for line in textFile:

        l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B = line.split()

        l0_temp_A = float(l0_temp_A)
        l1_temp_A = float(l1_temp_A)

        l0_temp_B = float(l0_temp_B)
        l1_temp_B = float(l1_temp_B)

        print([l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B])

        if l1_temp_A > l0_temp_A:
            M_temp_A = 1
        else:
            M_temp_A = 0

        if l1_temp_B > l0_temp_B:
            M_temp_B = 1
        else:
            M_temp_B = 0

        M1_A = np.append(M1_A, M_temp_A)
        M1_B = np.append(M1_B, M_temp_B)


with open("M0_AB_test_result_09262018.txt") as textFile:
    for line in textFile:

        l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B = line.split()

        l0_temp_A = float(l0_temp_A)
        l1_temp_A = float(l1_temp_A)

        l0_temp_B = float(l0_temp_B)
        l1_temp_B = float(l1_temp_B)

        print([l0_temp_A, l1_temp_A, l0_temp_B, l1_temp_B])

        if l1_temp_A > l0_temp_A:
            M_temp_A = 1
        else:
            M_temp_A = 0

        if l1_temp_B > l0_temp_B:
            M_temp_B = 1
        else:
            M_temp_B = 0

        M0_A = np.append(M0_A, M_temp_A)
        M0_B = np.append(M0_B, M_temp_B)


f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True,figsize=(3.39,2.3))
bins = [-0.01, 0.4, 0.6,  1.01]
fontdict = {'fontsize': 7}

left=-0.166
right=1.166
bottom=0
top=100

ax1.set_xlim(left=left,right=right)
ax2.set_xlim(left=left,right=right)


ax1.set_ylim(bottom=bottom, top=top)
ax2.set_ylim(bottom=bottom, top=top)


xticks = [0.2,0.8]
xticklabels = [r"$\mathcal{M}_0$",r"$\mathcal{M}_1$"]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels, fontdict=fontdict)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels, fontdict=fontdict)



yticks = [0,25,50,75,100]
ax1.set_yticks(yticks)
ax2.set_yticks(yticks)


ax1.set_yticklabels(yticks,fontdict=fontdict)
ax2.set_yticklabels(yticks,fontdict=fontdict)


axis_width=0.6
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(axis_width)
  ax2.spines[axis].set_linewidth(axis_width)


ax1.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax2.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)


ax1.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax2.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)



ax1.hist([M2_A,M1_A,M0_A], bins=bins, color=["xkcd:blue", "xkcd:green", "xkcd:red"], label=
         [r"$A$ affects $B$, $B$ affects $A$", r"$A$ affects $B$, $B$ doesn't affect $A$", r"$A$ and $B$ are independent"])


#ax1.grid(b=True, axis='y')

ax1.set_title("(a)", fontdict=fontdict)
#ax1.set_xlabel(r'$\mathcal{M}_2-A$', fontdict=fontdict)


ax2.hist([M2_B,M1_B,M0_B], bins=bins, color=["xkcd:blue", "xkcd:green", "xkcd:red"])

ax2.set_title("(b)", fontdict=fontdict)
#ax2.set_xlabel(r'$\mathcal{M}_2-B$', fontdict=fontdict)

ax1.legend(loc='upper center', bbox_to_anchor=(0., -0.03, 2., -0.1), borderaxespad=0.5,frameon=False, fontsize=7, ncol=1, mode="expand")
plt.subplots_adjust(wspace=0.3, hspace=0.6)
#plt.tight_layout()

plt.savefig("M210_hist_v2.pdf",bbox_inches='tight')













