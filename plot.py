import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from hyperparameters import HyperparameterSet
from PIL import Image

"""
Functions for plotting
"""

#plots everything
def drawMainWindow(xMin, xMax, yMin, yMax, evs, stdevs):
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'green', 'yellow', 'red'], 256)

	plt.subplot(1, 2, 1)
	plt.title("Predicted Temperature")
	# tell imshow about color map so that only set colors are used
	img = plt.imshow(evs, interpolation='nearest', cmap = cmap, origin = 'lower', extent = [xMin, xMax, yMin, yMax])
	# make a color bar
	plt.colorbar(img, cmap=cmap)
	plt.ylabel("latitude")
	plt.xlabel("longitude")

	plt.subplot(1, 2, 2)
	plt.title("Standard Deviation")
	img2 = plt.imshow(stdevs, interpolation='nearest', cmap = cmap, origin = 'lower', extent = [xMin, xMax, yMin, yMax])
	plt.colorbar(img2, cmap = cmap)
	plt.ylabel("latitude")
	plt.xlabel("longitude")

	plt.show()

def drawStations(xMin, xMax, yMin, yMax, stations):
	plt.title("Background Map")
	img3 = plt.imread("imgs/madagascar.png")
	plt.scatter(map(lambda x: x.loc[0], filter(lambda x: x.temperature != "", stations)), map(lambda x: x.loc[1], filter(lambda x: x.temperature != "", stations)), s=40)
	plt.ylabel("latitude")
	plt.xlabel("longitude")
	plt.imshow(img3, cmap = 'gray', extent = [xMin, xMax, yMin, yMax])

	plt.show()

#plot reward function vs. possible values of one hyperparameter
#DOES NOT CURRENTLY WORK: need to modify the line "y.append(objFunction..." since those aren't the right parameters
def drawHPSpace2D(data_points, hpValues, objFunction, kernel):
	#plt.subplot(2, 2, 1)
	plt.title("Objective Function Values")
	y = []
	for value in hpValues:
		y.append(objFunction(data_points, kernel, value))
	plt.scatter(hpValues, y)

	plt.show()

#plot the objective function (ex: sum of log likelihood) as a function of hyperparameters (length/kernel scale)
def drawHPSpace3D(data_points, objFunction, min_len, max_len, min_kernel, max_kernel, kernel, resH = 20, resV = 20):
	plt.title("Objective Function Values")
	len_increment = (max_len - min_len) / resH
	kernel_increment = (max_kernel - min_kernel) / resV

	func_values = [[0 for j in range(resH)] for i in range(resV)]

	for i in range(resV):
		for j in range(resH):
			hpset = HyperparameterSet(min_len + (j + 0.5) * len_increment, min_kernel + (i + 0.5) * kernel_increment)
			func_values[i][j] = objFunction(data_points, kernel, hpset.len_scale, hpset.kernel_scale)

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'green', 'yellow', 'red'], 256)

	img = plt.imshow(func_values, interpolation='nearest', cmap = cmap, origin = 'lower', extent = [min_len, max_len, min_kernel, max_kernel])
	plt.colorbar(img, cmap=cmap)
	plt.ylabel("Length Scale")
	plt.xlabel("Kernel Scale")
	plt.show()