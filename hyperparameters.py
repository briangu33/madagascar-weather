import predict
import math
from scipy import stats

"""
Optimize length scale, observation noise, and variance sill
"""
#TODO: Create hyperparameter set wrapper class which assigns coordintae values from hyperparameter space to property names
#makes this optimization code more extendable if you have a lot of hyperparameters

class HyperparameterSet:

	def __init__(self, len_scale, kernel_scale, obs_noise = 0.):
		self.len_scale = len_scale
		self.obs_noise = obs_noise
		self.kernel_scale = kernel_scale

#reward function: log likelihood of one observed value, given all the others as well as length scale, kernel scale, observation noise
def loocvLogLikelihood(data_points, left_out_point, kernel, len_scale, kernel_scale, obs_noise):
	copy_points = []
	for point in data_points:
		if point.temperature != '' and point.loc[0] != left_out_point.loc[0] and point.loc[1] != left_out_point.loc[1]:
			copy_points.append(point)

	predictedEV = predict.predictTemperatures([left_out_point.loc], copy_points, kernel, len_scale)[0]
	predictedStDev = predict.predictStdevs([left_out_point.loc], copy_points, kernel, len_scale, kernel_scale, obs_noise)[0]

	likelihood = stats.norm(predictedEV, predictedStDev).pdf(left_out_point.temperature)
	if likelihood == 0:
		return float("-inf")
	return math.log(likelihood)

#reward function: sum of log likelihoods over all points using the leave-one-out method (calls loocvLogLikelihood)
def loocvSumLogLikelihood(data_points, kernel, len_scale, kernel_scale, obs_noise = 0):
	sum = 0

	for point in data_points:
		if point.temperature != '':
			sum += loocvLogLikelihood(data_points, point, kernel, len_scale, obs_noise, kernel_scale)

	return sum
	

def optimizeHyperparameters(min_HPSet, max_HPSet, kernel, data_points, num_iterations, objFunc = loocvSumLogLikelihood, num_increments = 5):
	'''input: a bounding box for the hyperparameter search (given by the corners min_HPSet and max_HPSet), kernel, data points, 
	number of times to "zoom in" (num_iterations), the objective cost or reward function, and the number of ticks to divide search space into
	returns a set of hyperparameters determined to minimize cost function/maximize reward function'''

	best_obj_val = float("-inf")
	
	best_hyperparameters = HyperparameterSet((max_HPSet.len_scale + min_HPSet.len_scale) / 2, (max_HPSet.kernel_scale + min_HPSet.kernel_scale) / 2, (max_HPSet.obs_noise + min_HPSet.obs_noise) / 2)
	increment_len_scale = (max_HPSet.len_scale - min_HPSet.len_scale) / num_increments
	increment_obs_noise = (max_HPSet.obs_noise - min_HPSet.obs_noise) / num_increments
	increment_kernel_scale = (max_HPSet.kernel_scale - min_HPSet.kernel_scale) / num_increments

	for i in range(num_increments + 1):
		for j in range(1):
			for k in range(num_increments + 1):
				len_scale = min_HPSet.len_scale + i * increment_len_scale
				obs_noise = min_HPSet.obs_noise + j * increment_obs_noise
				kernel_scale = min_HPSet.kernel_scale + k * increment_kernel_scale

				current_obj_val = objFunc(data_points, kernel, len_scale, kernel_scale, obs_noise)
				if current_obj_val > best_obj_val:
					best_hyperparameters = HyperparameterSet(len_scale, kernel_scale, obs_noise)
					best_obj_val = current_obj_val

				#print len_scale, kernel_scale, obs_noise, current_obj_val

	if num_iterations == 1:
		return best_hyperparameters

	new_min_HPSet = HyperparameterSet(max(min_HPSet.len_scale, best_hyperparameters.len_scale - increment_len_scale), max(min_HPSet.kernel_scale, best_hyperparameters.kernel_scale - increment_kernel_scale), max(min_HPSet.obs_noise, best_hyperparameters.obs_noise - increment_obs_noise))
	new_max_HPSet = HyperparameterSet(min(max_HPSet.len_scale, best_hyperparameters.len_scale + increment_len_scale), min(max_HPSet.kernel_scale, best_hyperparameters.kernel_scale + increment_kernel_scale), min(max_HPSet.obs_noise, best_hyperparameters.obs_noise + increment_obs_noise))

	return optimizeHyperparameters(new_min_HPSet, new_max_HPSet, kernel, data_points, num_increments, objFunc, num_iterations - 1)

#The functions below don't work anymore
"""
def loocvSquaredError(data_points, left_out_point, kernel, len_scale):
	#create a copy of the data points without the left out point so you don't modify data_points
	copy_points = []
	for point in data_points:
		if point.temperature != '' and point.loc[0] != left_out_point.loc[0] and point.loc[1] != left_out_point.loc[1]:
			copy_points.append(point)

	predicted = predict.predictTemperatures([left_out_point.loc], copy_points, kernel, len_scale)[0]
	return (predicted - left_out_point.temperature) ** 2

def loocvNegMeanSquaredError(data_points, kernel, len_scale):
	mean = 0
	count = 0

	for point in data_points:
		if point.temperature != '':
			count += 1
			#data_points.remove(point)
			mean += loocvSquaredError(data_points, point, kernel, len_scale)
			#data_points.append(point)
	mean /= count
	#print len_scale, mean
	return -1*mean
"""