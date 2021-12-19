#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:10:39 2021

@author: alessio
"""

import random
import csv
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt

###############################################################################
#Generate the dataset
###############################################################################

#Universal gas constant
R = 8.314

#Number of instances of the dataset we want to generate
num_points = 5000

#Variables containing P,V,T of the dataset
V_list = []
P_list = []
T_list = []

#Generate V and P randomly in the intervals (2.5e-2 m^3,5e-2 m^3) and $(1e5 Pa K, 2e5 Pa)$
for i in range(num_points):
    temp_V = 2.5e-2+5e-2*random.random()
    temp_P = 1e5+1e5*random.random()
    V_list.append(temp_V)
    P_list.append(temp_P)

#Compute T from P and V
for i in range(num_points):
    temp_T = V_list[i]*P_list[i]/8.314
    T_list.append(temp_T)

###############################################################################
#Prepare the data for training
###############################################################################

#Normalize data and prepare data for training
V_max = np.max(V_list)
V_min = np.min(V_list)
T_max = np.max(T_list)
T_min = np.min(T_list)
P_max = np.max(P_list)
P_min = np.min(P_list)

in_points = np.array(list(zip((V_list-V_min)/(V_max-V_min),(P_list-P_min)/(P_max-P_min))))
out_points = np.array(list(zip((T_list-T_min)/(T_max-T_min))))
half = int(num_points/2)
in_train = in_points[0:half]
out_train = out_points[0:half]
in_test = in_points[half:num_points]
out_test = out_points[half:num_points]

###############################################################################
#Write dataset to file
###############################################################################

#Write dataset to file
with open('thermodynamic_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(num_points):
        new = np.concatenate([in_points[i],out_points[i]])
        csv_writer.writerow(new)   

###############################################################################
#Build and train the neural network
###############################################################################


#Build the neural network
model = Sequential()
model.add(Dense(5, input_shape=(2,),activation = 'sigmoid'))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(5,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='Adam')

#Train the model
history = model.fit(in_train, out_train, batch_size=1024, epochs=10000,
          verbose=1, validation_data=(in_test, out_test))


###############################################################################
#Extract the weights
###############################################################################

weights_list = []
biases_list = []
t_weights_list = []

nlayers = len(model.layers)

for i in range(nlayers):
    t_weights_list.append(model.layers[i].get_weights()[0])
    biases_list.append(model.layers[i].get_weights()[1])

#Tensorflow returns the transpose of the matrices of weights we need
for i in range(nlayers):
    weights_list.append(np.transpose(t_weights_list[i]))

###############################################################################
#Get the activation function of the layers
###############################################################################

layers_act = []

for i in range(nlayers):
    act_fun = model.layers[i].get_config()['activation']
    if (act_fun == 'sigmoid'):
        layers_act.append("FC_LAYER_SG")
    if (act_fun == 'softmax'):
        layers_act.append("FC_LAYER_SM")
    if (act_fun == 'sofplus'):
        layers_act.append("FC_LAYER_SP")

###############################################################################
#Write the weights to file
###############################################################################

temp = np.ndarray([0])

with open('weights.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([nlayers])    
    for i in range(nlayers):
        csv_writer.writerow([len(model.layers[i].get_weights()[1]),len(model.layers[i].get_weights()[0]),len(model.layers[i].get_weights()[1]),len(model.layers[i].get_weights()[0]),layers_act[i]])
        for j in range(len(weights_list[i])):
            temp = np.concatenate([temp,weights_list[i][j]])
        csv_writer.writerow(temp)
        temp = np.ndarray([0])
        csv_writer.writerow(biases_list[i])
   
    weights_list.append(model.layers[i].get_weights()[0])

###############################################################################
#Plot the dataset
###############################################################################

nV = (V_list-V_min)/(V_max-V_min)
nP = (P_list-P_min)/(P_max-P_min)
plt.scatter(nV,nP, alpha=0.5)
plt.show()
plt.clf()


print("V Max/min: ", V_max, " / ", V_min)
print("P Max/min: ", P_max, " / ", P_min)


