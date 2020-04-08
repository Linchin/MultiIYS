__author__ = "Lingqing Gan"

# ------------------------------------------------------------------------------------

#   FILE_NAME:      legc06_YSP_inter_bi_macro.py

#   DESCRIPTION:    Gibbs sampling for single variable time series state
#                   transition and precision (variance).
#                   Using Asher's dissertation chapter 2.

#                   Adding: two variables where one is influencing another


#   09/17/2018     a macro file to run legc02_YSP_inter_bi.py multiple times and record the results


#   AUTHOR:         Lingqing Gan (Stony Brook University)

#   DATE:           09/17/2018 - 09/17/2018

# ------------------------------------------------------------------------------------

import datetime
import legc02_YSP_inter_bi

print("start time:")
print(datetime.datetime.now())

with open('M1_10_test_result_09172018.txt', 'w') as f:
    for i in range(0, 100):
        temp1, temp2, temp3, temp4 = legc02_YSP_inter_bi.main()
        f.write("{} {} {} {} \n".format(temp1, temp2, temp3, temp4))


print("end time: ")
print(datetime.datetime.now())