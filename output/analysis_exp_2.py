#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:08:02 2020

@author: alessio
"""

# imports for array-handling and plotting
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import csv

#List to save the rows of the csv file
data = []

#Labels of the plots
labels = ["x","y"]

#Csv file containing the data (left part of the e.c.)
data_file = "surface_2_simec_out_projection.csv"

#Var for the number of points in the csv file
num_points = 0

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_points += 1

#To build both the sides of the class of equivalence, we need to run the algorithm
#in both direction.
'''
#Csv file containing the data (left part of the e.c.)
data_file = "surface_2_simec_out_left.csv"

#Var for the number of points in the csv file
num_points = 0

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_points += 1

#Csv file containing the data (right part of the e.c.)
data_file = "surface_2_simec_out_right.csv"

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_points += 1
'''

#Lists containing v and p of the equivalence class
x_data = []
y_data = []

#Save x,y coordinates of the equivalence class in x_data and y_data
for i in range(num_points):
    x_data.append(data[i][0])
    y_data.append(data[i][1])

#Plot the equivalence class
fig, ax = plt.subplots()
plt.scatter(x_data,y_data, color='blue', zorder=0)

rec = plt.Rectangle((-1,-1), 2, 2, facecolor="C1", alpha=0.5, zorder=-1)
ax.add_patch(rec)


x_level_set = np.linspace(0.25, 2, 1000)
y_level_set = -1.*x_level_set*x_level_set+.3125#.1875
plt.scatter(x_level_set,y_level_set, color='red',zorder=1, s=2)

plt.gca().set_aspect('equal', adjustable='box')
'''
plt.xlim([0,2])
plt.ylim([-6,1])
'''
plt.xlim([.75,1.75])
plt.ylim([-1.5,-.5])
plt.savefig("plot.png")
plt.clf()
