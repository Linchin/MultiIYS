# coding: utf-8

"""
Date: 2019.09.18

Title: exp12-calculate_likelihood.py

Description:
After saving the original signal in a txt file,
for each node pair,
we calculate the distance between their 1s.
then we plot the histogram of the distances.
And calculate the likelihood based on the distance.
"""
from matplotlib import pyplot as plt


# read the file and save them in arrays
signal = []
node_number = 3
for i in range(node_number):
    signal.append([])

with open("exp07-data-20190906-134627-signals.txt", 'r') as file:
    for line in file:
        for i in range(0, node_number):
            signal[i].append(line.split(" ")[i])

time_slots = len(signal[0])

# calculate the distances for each possible node combo
distance = []

for i in range(node_number):
    distance.append([])
    for j in range(node_number):
        distance[i].append([])

for i in range(node_number):
    # index of influencing node
    for j in range(node_number):
        # index of influenced node
        ti = 0
        tj = 0
        for ii in range(time_slots):
            if signal[i][ii] == '1':
                ti = ii
            if signal[j][ii] == '1':
                distance[i][j].append(ii - ti)

print("A")

# plot the histogram
# generate subplots
fig, axes = plt.subplots(node_number,
                         node_number,
                         sharex=True,
                         sharey=True)

for i in range(0, node_number):
    for j in range(0, node_number):
        axes[i, j].hist(distance[i][j])
        axes[i,j].title.set_text(str((i,j)))

plt.show()

# calculate the likelihood









