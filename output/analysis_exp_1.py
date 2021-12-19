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

#Csv file containing the data
data_file = "surface_1_bad_simec_out.csv"

#Var for the number of points in the csv file
num_points = 0

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_points += 1

#Lists containing v and p of the equivalence class
x_data = []
y_data = []

#Save x,y coordinates of the equivalence class in x_data and y_data
for i in range(num_points):
    x_data.append(data[i][0])
    y_data.append(data[i][1])

#Plot the equivalence class
fig, ax = plt.subplots()

plt.scatter(x_data,y_data)
'''
for j in range(num_points):
    plt.plot(row[0],row[1], 'o', color='black');
            
            #plt.imshow(narr.reshape(2,2), cmap='gray', interpolation='none')
            #plt.xticks([])
            #plt.yticks([])
            #plt.savefig(str(j)+".png")
            #plt.clf()
'''         

circ = plt.Circle((0, 0), .353, color='red', fill = False, alpha=0.5, linewidth=4)
ax.add_patch(circ)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-.7,.7])
plt.ylim([-.7,.7])
plt.savefig("Complete.png")
plt.clf()
