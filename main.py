import math
from scipy import linalg

import getinput
import predict
import plot
from station import Station
from hyperparameters import optimizeHyperparameters
from hyperparameters import loocvSumLogLikelihood
from hyperparameters import HyperparameterSet

#choose the day you want to plot
day = 1 

#define the boundaries of the visualization
min_lat = -27.0
max_lat = -11.0
min_long = 42.0
max_long = 51.0

#choose the hyperparameter defaults: length scale, kernel scale (sigma_f squared), observation noise (variance)
hpset = HyperparameterSet(2., 1., 0.)

#choose kernel
kernel = predict.gaussianKernel

#define horizontal and vertical resolution of visualization
resH = 100
resV = 100

"""
Reading the input, generating predictions, displaying predictions.
"""

#open excel sheet, get data
stations = getinput.getStations(day)
getinput.printData(stations)

#plot station locations
plot.drawStations(min_long, max_long, min_lat, max_lat, stations)

#optimize hyperparameter (length scale)
min_HPSet = HyperparameterSet(0.1, 0.1, 0.)
max_HPSet = HyperparameterSet(10., 10., 0.)
hpset = optimizeHyperparameters(min_HPSet, max_HPSet, kernel, stations, 7)
print hpset.len_scale, hpset.kernel_scale, hpset.obs_noise

#initialize and fill in the temperature prediction grid (resH by resV grid of temperatures)
prediction = predict.getPredictions(resV, resH, min_lat, max_lat, min_long, max_long, kernel, stations, hpset.len_scale, hpset.kernel_scale, hpset.obs_noise)

#plot the predictions
plot.drawMainWindow(min_long, max_long, min_lat, max_lat, prediction.temperatures, prediction.stdevs)

#plot the objective function vs. hyperparameters graph
plot.drawHPSpace3D(stations, loocvSumLogLikelihood, 1., 3., 1., 3.5, kernel, 40, 40)