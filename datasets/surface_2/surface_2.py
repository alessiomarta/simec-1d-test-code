#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:27:05 2020

@author: alessio
"""
# imports for array-handling and plotting
import numpy as np
import random
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import csv

# keras imports for the dataset and building our neural network
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
import tensorflow as tf
from keras.layers import Softmax
from mpl_toolkits.mplot3d import Axes3D
import math

###############################################################################
#Generate the dataset
###############################################################################

#Lists for build the dataset
xpts = []
ypts = []
zpts = []
points = []

#Number of points to generate
num_points = 2000

#Generate the points on the surface z = x^2 + y^2 - 2 in (-1,1)x(-1,1)
for i in range(num_points):
    xp = 2.*(random.random()-.5)
    yp = 2.*(random.random()-.5)
    zp = math.exp(xp*xp+yp-2)
    xpts.append(xp)
    ypts.append(yp)
    zpts.append(zp)
    points.append([xp,yp,zp])

#Plot the cloud of points of the surface
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xpts, ypts, zpts)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.clf()

###############################################################################
#Write the dataset to file
###############################################################################

#Write the dataset to file
with open('surface.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(num_points):
        csv_writer.writerow(points[i])

###############################################################################
#Prepare data for training
###############################################################################

#Prepare data for training
in_points = np.array(list(zip(xpts,ypts)))
out_points = np.array(zpts)
half = int(num_points/2)
in_train = in_points[0:half]
out_train = out_points[0:half]
in_test = in_points[half:num_points]
out_test = out_points[half:num_points]

###############################################################################
#Build and train  the neural network
###############################################################################

#Build the neural network
model = Sequential()
model.add(Dense(5, input_shape=(2,),activation = 'sigmoid'))
model.add(Dense(5,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

#Train the model
history = model.fit(in_train, out_train, batch_size=512, epochs=20000,
          verbose=2, validation_data=(in_test, out_test))

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
#Plot the image of the neural network over (-1,1)x(-1,1)
###############################################################################

#Set lenght of the grid
len_x_grid = 20
len_y_grid = 20

#Generate x-y grid
x_grid = np.linspace(-1, 1, len_x_grid)
y_grid = np.linspace(-1, 1, len_y_grid)
grid = np.transpose([np.tile(x_grid, len_y_grid), np.repeat(y_grid, len_x_grid)])

#Compute image of the grid through the nn map 
grid_nn_output = model.predict(grid)

#Vars in which we save the cloud of points built out from the grid
points_x = []
points_y  = []
points_z  = []

for i in range(len_x_grid):
    for j in range(len_y_grid):
        points_x.append(x_grid[i])
        points_y.append(y_grid[j])
        points_z.append(grid_nn_output[len_x_grid*j+i][0])
   

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(points_x, points_y, points_z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
