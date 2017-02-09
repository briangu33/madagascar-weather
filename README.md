# madagascar-weather

This program implements [Gaussian Process Regression](http://www.gaussianprocess.org/gpml/chapters/) and uses it to predict temperatures across Madagascar based on local weather station data. It also performs basic hyperparameter optimization using leave-one-out cross validation. It was written in 2015 as a first Python project.

To display station maps, temperature prediction maps, and a visualization of hyperparameter performance, simply run `python main.py`.

# Dataset
`data.xls` contains temperature data from various weather stations across Madagascar over a 365-day period. A README for the original source of the data can be found [here](https://www1.ncdc.noaa.gov/pub/data/gsod/readme.txt). The data is available under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

The data file specifies locations of 26 stations. Not all stations have temperature data for all days. By default, data and predictions are plotted for day 1; a specific day can be chosen by modifying the `day` parameter in `main.py`. Locations of stations with usable data are initially displayed when the program is run. Below, we show station locations for day 1:

![day 1 station locations](/imgs/d1stations.png)

# Temperature Prediction

Given temperature data and a set of hyperparameters (length scale, observation noise, kernel scale), `madagascar-weather` can predict temperatures and uncertainties at arbitrary points across Madagascar. The variance of a prediction at a point increases as the distance between the point and nearby weather stations increases. 

Predictions (mean and standard deviation) are visualized in a heatmap. The prediction heatmaps for day 1 are shown below.

![day 1 predictions](/imgs/temp_predict.png)

By default, a Gaussian kernel function is used to generate covariance matrices. Other kernels (tricube kernel, etc.) can also be used easily; kernel functions are implemented in `predict.py`, and specified in `main.py`. 

Hyperparameters can either be directly specified, or else automatically chosen through the built-in hyperparameter optimization functions (see below). 

# Hyperparameter Optimization

Hyperparameters can be automatically selected through leave-one-out cross validation (LOOCV). The objective function we maximize is the average log-likelihood of seeing any given data point, given the remainder of the data and the chosen hyperparameters. This is done with a fairly basic approach: we calculate the value of the objective function at various points in hyperparameter space, throw out regions of the space that perform poorly, and zoom in on plausible regions. This idea is explored in further detail in the `toy-gpr` repository, where Gaussian Process Regression is used to select hyperparameters (whoa, meta).

For visualization purposes, we can graph the value of the objective function (average log likelihood of the data according to LOOCV) against kernel and length scale. The objective function appears to be nicely convex in these two hyperparameters.

![objective function values](/imgs/hp_opt.png)

# Acknowledgements

This project was done during a part-time data internship at the [Institute for Disease Modeling](http://idmod.org/) under the mentorship of Daniel Klein. 
