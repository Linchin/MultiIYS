__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      plot05-histogram_of_results_2.py

#   DESCRIPTION:    reading the results produced by YSP_inter_bi_bidir_macro.py
#                   and draw histogram(s) with it.
#


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/27/2018 - 09/27/2018

#   UPDATE 01:      10/25/2018
#                   improved format for icassp

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


f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True,figsize=(3.39,2.3))
bins = [-0.166, 0.166, 0.5, 0.833, 1.166]
fontdict = {'fontsize': 7}

left=-0.166
right=1.166
bottom=0
top=100

ax1.set_xlim(left=left,right=right)
ax2.set_xlim(left=left,right=right)
ax3.set_xlim(left=left,right=right)
ax4.set_xlim(left=left,right=right)
ax5.set_xlim(left=left,right=right)
ax6.set_xlim(left=left,right=right)

ax1.set_ylim(bottom=bottom, top=top)
ax2.set_ylim(bottom=bottom, top=top)
ax3.set_ylim(bottom=bottom, top=top)
ax4.set_ylim(bottom=bottom, top=top)
ax5.set_ylim(bottom=bottom, top=top)
ax6.set_ylim(bottom=bottom, top=top)

xticks = [0,1]
xticklabels = [r"$\mathcal{M}_0$",r"$\mathcal{M}_1$"]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels, fontdict=fontdict)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels, fontdict=fontdict)
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticklabels, fontdict=fontdict)
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticklabels, fontdict=fontdict)
ax5.set_xticks(xticks)
ax5.set_xticklabels(xticklabels, fontdict=fontdict)
ax6.set_xticks(xticks)
ax6.set_xticklabels(xticklabels, fontdict=fontdict)

yticks = [0,50,100]
ax1.set_yticks(yticks)
ax2.set_yticks(yticks)
ax3.set_yticks(yticks)
ax4.set_yticks(yticks)
ax5.set_yticks(yticks)
ax6.set_yticks(yticks)

ax1.set_yticklabels(yticks,fontdict=fontdict)
ax2.set_yticklabels(yticks,fontdict=fontdict)
ax3.set_yticklabels(yticks,fontdict=fontdict)
ax4.set_yticklabels(yticks,fontdict=fontdict)
ax5.set_yticklabels(yticks,fontdict=fontdict)
ax6.set_yticklabels(yticks,fontdict=fontdict)

axis_width=0.6
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(axis_width)
  ax2.spines[axis].set_linewidth(axis_width)
  ax3.spines[axis].set_linewidth(axis_width)
  ax4.spines[axis].set_linewidth(axis_width)
  ax5.spines[axis].set_linewidth(axis_width)
  ax6.spines[axis].set_linewidth(axis_width)

ax1.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax2.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax3.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax4.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax5.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)
ax6.tick_params(axis='y', direction = 'in', length = 2, width = axis_width)


ax1.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax2.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax3.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax4.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax5.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)
ax6.tick_params(axis='x', direction = 'in', length = 0, width = axis_width)


ax1.hist(M2_A, bins=bins, color="xkcd:blue")

#ax1.grid(b=True, axis='y')

ax1.set_title("(a) $\mathcal{M}_2-A$", fontdict=fontdict)
#ax1.set_xlabel(r'$\mathcal{M}_2-A$', fontdict=fontdict)


ax2.hist(M2_B, bins=bins, color="xkcd:blue")

ax2.set_title("(b) $\mathcal{M}_2-B$", fontdict=fontdict)
#ax2.set_xlabel(r'$\mathcal{M}_2-B$', fontdict=fontdict)


ax3.hist(M1_A, bins=bins, color="xkcd:blue")

ax3.set_title("(c) $\mathcal{M}_1-A$", fontdict=fontdict)
#ax3.set_xlabel(r'$\mathcal{M}_1-A$', fontdict=fontdict)


ax4.hist(M1_B, bins=bins, color="xkcd:blue")

ax4.set_title("(d) $\mathcal{M}_1-B$", fontdict=fontdict)
#ax4.set_xlabel(r'$\mathcal{M}_1-B$', fontdict=fontdict)


ax5.hist(M0_A, bins=bins, color="xkcd:blue")
ax5.set_title("(e) $\mathcal{M}_0-A$", fontdict=fontdict)
#ax5.set_xlabel(r'$\mathcal{M}_0-A$', fontdict=fontdict)


ax6.hist(M0_B, bins=bins, color="xkcd:blue")
ax6.set_title("(f) $\mathcal{M}_0-B$", fontdict=fontdict)
#ax6.set_xlabel(r'$\mathcal{M}_0-B$', fontdict=fontdict)

plt.subplots_adjust(wspace=0.3, hspace=0.6)
#plt.tight_layout()

plt.savefig("M210_hist.pdf",bbox_inches='tight')













