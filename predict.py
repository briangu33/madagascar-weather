import math
import numpy as np
import getinput
from hyperparameters import optimizeHyperparameters

class Prediction:

	def __init__(self, temperatures, stdevs):
		self.temperatures = temperatures
		self.stdevs = stdevs

"""
Functions for generating the temperature predictions
"""

#distance function between two vectors x1 and x2
def dist(x1, x2):
	squared_dist = 0.0
	for i in range(len(x1)):
		squared_dist += (x1[i] - x2[i])**2
	return squared_dist**0.5

#gaussian kernel; returns appropriate weight for x2 on x1 and vice versa
def gaussianKernel(x1, x2, len_scale):
	return math.exp(-1. * ((dist(x1, x2)/len_scale)**2.) / 2.)

#tricube kernel
def tricubeKernel(x1, x2, len_scale):
	d = dist(x1, x2) / len_scale
	if d <= 1:
		return (1 - d ** 3) ** 3
	return 0

#takes in two sets of points as input (X and Xstar) and returns covariances between all (x, x*) pairs in a matrix
def covMatrix(Xstar, X, kernel, len_scale):
	mat = [[0 for i in range(len(X))] for j in range(len(Xstar))]
	for j in range(len(Xstar)):
		for i in range(len(X)):
			mat[j][i] = kernel(X[i], Xstar[j], len_scale)
	return np.array(mat)

#given the data, kernel, length scale: predict temperature at locations in Xstar
def predictTemperatures(Xstar, data_points, kernel, len_scale):
	X = []
	trainingTemp = []
	mean = 0
	numPoints = 0

	#formulas only valid under a 0-centered prior, so we subtract the mean of the data from all points
	for point in data_points:
		if point.temperature != '':
			mean += point.temperature
			numPoints += 1
	mean /= numPoints

	for point in data_points:
		if point.temperature != '':
			X.append(point.loc)
			trainingTemp.append([point.temperature])

	for temp in trainingTemp:
		temp[0] -= mean

	temperatures = np.dot(np.dot(covMatrix(Xstar, X, kernel, len_scale), np.linalg.inv(covMatrix(X, X, kernel, len_scale))), np.array(trainingTemp))

	result = []
	for arr in temperatures: #since temperatures is the result of np.dot, temperatures is an array of 1-length arrays
		result.append(arr.tolist()[0] + mean) #normalize back to be centered around the mean
	return result

#given the data, kernel, length scale, kernel scale: give the standard deviations in temperature predictions at locations in Xstar
def predictStdevs(Xstar, data_points, kernel, len_scale, obs_noise, kernel_scale):
	X = []

	for point in data_points:
		if point.temperature != '':
			X.append(point.loc)

	#we only care about diagonal entries, so we don't actually need to calculate covMatrix(Xstar, Xstar)
	#this saves time and we can easily account for it later by replacing variances[i][i] with 1 - oneMinusVariances[i][i]
	#note that oneMinusVariances[i][j] is meaningless though when i != j
	oneMinusVariances = np.dot(np.dot(covMatrix(Xstar, X, kernel, len_scale), np.linalg.inv(covMatrix(X, X, kernel, len_scale) + obs_noise * np.identity(len(X)))), covMatrix(X, Xstar, kernel, len_scale))

	stdevs = []
	for i in range(len(oneMinusVariances)):
		stdevs.append(kernel_scale * ((1 - oneMinusVariances[i][i]) ** 0.5))
	return stdevs

#returns the temperature predictions for each cell. resolution, bounds, kernel type, and data can be changed at will
def getPredictions(resV, resH, min_lat, max_lat, min_long, max_long, kernel, data_points, len_scale, kernel_scale, obs_noise):
	tempPredictions = [[0 for x in range(resH)] for y in range(resV)]
	stdevPredictions = [[0 for x in range(resH)] for y in range(resV)]

	locations = [[0, 0] for i in range(resV * resH)]

	#this for loop will do test = training from the data (use only for testing)
	#for point in data_points:
	#	if point.temperature != '':
	#		curr_x = int((point.loc[0] - min_long) / (max_long - min_long) * resH)
	#		curr_y = int((point.loc[1] - min_lat) / (max_lat - min_lat) * resV)
	#		locations[curr_y * resH + curr_x] = [point.loc[0], point.loc[1]]

	for y in range(resV):
		for x in range(resH):
			#map position in array to a location
			curr_lat = ((y + 0.5) * (max_lat - min_lat) / resV) + min_lat
			curr_long = ((x + 0.5) * (max_long - min_long) / resH) + min_long
			locations[y * resH + x] = [curr_long, curr_lat]

	#fill in the temperature prediction, using kernel of your choice
	temperatures = predictTemperatures(locations, data_points, kernel, len_scale)
	stdevs = predictStdevs(locations, data_points, kernel, len_scale, obs_noise, kernel_scale)
	for y in range(resV):
		for x in range(resH):
			tempPredictions[y][x] = temperatures[y * resH + x]
			stdevPredictions[y][x] = stdevs[y * resH + x]

	prediction = Prediction(tempPredictions, stdevPredictions)

	return prediction


#uses kernel regression to predict temperature for a given location
#def predictTemperature(loc, kernel, data_points):
#	weightedSum = 0.0
#	sumOfWeights = 0.0
#
#	for point in data_points:
#		if point.temperature != '':
#			weightedSum += point.temperature * kernel(loc, point.loc, 1.2) #I'm not actually sure what len_scale should be, but 1-1.5 seem to work best for Gaussian
#			sumOfWeights += kernel(loc, point.loc, 1.2)
#
#	return weightedSum / sumOfWeights