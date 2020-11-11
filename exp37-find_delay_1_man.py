# coding: utf-8

"""
file name:
exp37-find_delay_1_man.py

10/28/2020

Read the mat file that saves all the event time stamp.
Manually find the beginning and ending time stamps for
the delay1 sequences.
This is the manual initial version.
"""


import scipy.io as sio
import os
from os.path import dirname, join as pjoin
import numpy as np
import math

# load the event data matrix

print(dirname)

dir_path = dirname(os.path.realpath(__file__))
print(dir_path)

events_0530_name = dir_path + "\spike_data\M_20180530_merge_events.mat"
events_0530 = sio.loadmat(events_0530_name)

print("lol")