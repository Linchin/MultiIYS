__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      legc04_YSP_inter_bi_bidir_macro.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.

#                   Adding: two variables where one is influencing another


#   09/17/2018     a macro file to run legc03_YSP_inter_bi_bidir.py multiple times and record the results


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/26/2018 - 09/26/2018

# ------------------------------------------------------------------------------------

import datetime
import legc03_YSP_inter_bi_bidir

print("start time:")
print(datetime.datetime.now())

with open('M0_AB_test_result_09262018.txt', 'w') as f:
    for i in range(0, 100):
        temp1, temp2, temp3, temp4 = legc03_YSP_inter_bi_bidir.main()
        f.write("{} {} {} {} \n".format(temp1, temp2, temp3, temp4))


print("end time: ")
print(datetime.datetime.now())