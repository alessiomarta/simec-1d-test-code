#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:08:02 2020

@author: alessio
"""

# imports for array-handling and plotting
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import csv

#List to save the rows of the csv file
data = []

#Labels of the plots
labels = ["Volume","Pressure"]

#Csv file containing the data
data_file = "thermo_simec_out.csv"

#Var for the number of points in the csv file
num_data = 0

#Universal gas constant
R = 8.31

#Read the csv file
with open(data_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
        num_data += 1

#Lists containing v and p of the equivalence class
v_data = []
p_data = []

#Max/min values of the dataset
V_Max = 0.075
V_min = 0.025
P_Max = 2e5
P_min = 1e5

#Save v and p of the equivalence class in v_data and p_data, rescaling back to
#original (non-normalized) values
for i in range(num_data):
    v_data.append(V_min+data[i][0]*(V_Max-V_min))
    p_data.append(P_min+data[i][1]*(P_Max-P_min))

#Starting and ending points of the equivalence class
V_start = v_data[0]
V_end = v_data[num_data-1]
P_start = p_data[0]
P_end = p_data[num_data-1]

#Compute the temperature of the starting point, for comparison with the
#true isothermal curve
T_start = V_start*P_start/R
print("Temperature : ", T_start)

#Generate the true isothermal curve
n_approx = 1000
V_isotermal = np.linspace(V_start, V_end, n_approx)
P_isotermal = np.zeros(n_approx)
for i in range(n_approx):
    P_isotermal[i] = R*T_start/V_isotermal[i]

#Plot the two curves: eq. class (blues) and true isothermal curve (red)
plt.scatter(v_data, p_data, alpha=0.5, color='blue')
xa =  V_isotermal
ya = R*T_start/xa
plt.plot(xa, ya, color='red', linewidth=4, linestyle=':')
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.ticklabel_format(axis="x", style="sci",scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.show()
plt.clf()

#Print starting and final (P,V)
print("First V : ", scatter_data_0[0])
print("Fisrt P : ", scatter_data_1[0])
print("Final V : ", scatter_data_0[num_data-1])
print("Final P : ", scatter_data_1[num_data-1])


