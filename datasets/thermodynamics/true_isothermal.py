#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:06:55 2021

@author: alessio
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:10:39 2021

@author: alessio
"""

import random
import csv
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation

#Value (in Joule) of nRT For 1 mole and T=293.15 k.
C = 8.314*510

#Max/min values from numerical experiment
V_Max = 4.5e-4
V_min = 1.85e-4
P_Max = 22196822.98995688
P_min = 744142.2406089753
delta = 1e-7

#Number of instances of the dataset we want to generate
num_points = int((V_Max-V_min)/delta)
nn = 257

V_list = []
T = 510

for i in range(num_points):
    temp_vol = V_min+i*delta
    V_list.append(temp_vol)

a = 2.0158
b = 0.0001263

a = 1.602
b = .0001124
#Compute P
P_list = []
for i in range(num_points):
    V = V_list[i]
    C = 8.31*T
    P = C/(V-b)-a/(V*V)
    P_list.append(P)

#Normalize data and prepare data for training
V_max = np.max(V_list)
V_min = np.min(V_list)
P_max = np.max(P_list)
P_min = np.min(P_list)

print("V M/m: ", V_max, " -- ", V_min)
print("P M/m: ", P_max, " -- ", P_min)

import matplotlib.pyplot as plt
#nV = (V_list-V_min)/(V_max-V_min)
#nP = (P_list-P_min)/(P_max-P_min)
labels = ["Volume","Pressure"]
plt.xlabel(labels[0])
plt.ylabel(labels[1])
#plt.scatter(nV,nP, alpha=0.5)
plt.ticklabel_format(axis="x", style="sci",scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.scatter(V_list,P_list, alpha=0.5)
plt.show()
plt.clf()


