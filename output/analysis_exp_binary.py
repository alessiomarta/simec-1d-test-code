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

#Var for the number of points in the csv files
num_points = 0

#Csv file containing the data
data_file = "test/simexp_class_out.csv"

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_points += 1

#Lists containing v and y of the equivalence class
x_data = []
y_data = []

#Save x,y coordinates of the equivalence class in x_data and y_data
for i in range(num_points):
    x_data.append(data[i][0])
    y_data.append(data[i][1])

#Plot the equivalence class
fig, ax = plt.subplots()

plt.scatter(x_data,y_data,s=2,alpha = 0.5)

'''
rec = plt.Rectangle((-3.14,-1), 6.28, 2, fill=None, alpha=0.5)
ax.add_patch(rec)
'''

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-3.14,3.14])
plt.ylim([-1.,1.5])
plt.savefig("Complete.png")
plt.clf()
