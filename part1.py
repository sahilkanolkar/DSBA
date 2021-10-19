import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def plotData(x, y1, y2):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x, y1)
    ax[1].scatter(x, y2)
    plt.show()

def minimizeError(x, y, learning_rate):
    max_iterations = 100000
    slope = 0
    intercept = 0

    error = 0
    old_error = 0
    tolerance = 0.0000000001
    residual = 1

    while(residual > tolerance):
        y_predicted = slope * x + intercept
        derivate_by_slope = (-2) * sum(x * (y - y_predicted))
        derivate_by_intercept = (-2) * sum((y - y_predicted))

        slope -= learning_rate * derivate_by_slope
        intercept -= learning_rate * derivate_by_intercept

        error = sum(y - y_predicted)
        residual = abs(error - old_error)
        old_error = error
        max_iterations -= 1
        if max_iterations == 0:
            break

    return (slope,intercept,error)

def plotRegression(x, y, learning_rate):
    model = minimizeError(x, y, learning_rate)
    slope = model[0]
    intercept = model[1]
    error = model[2]
    y_predicted = slope * x + intercept

    plt.scatter(x, y)
    plt.plot(x,y_predicted, color = 'red')
    plt.show()

def predict(x):
    x_scaled = (x - x_year_mean)/x_year_std

    model_count = minimizeError(x_year_scaled, y_count_scaled, learning_rate_count)
    slope_count = model_count[0]
    intercept_count = model_count[1]

    y_count_pred_scaled = slope_count * x_scaled + intercept_count
    y_count_predicted = np.exp((y_count_pred_scaled * y_count_std) + y_count_mean)

    model_size = minimizeError(x_year_scaled, y_size_scaled, learning_rate_size)
    slope_size = model_size[0]
    intercept_size = model_size[1]

    y_size_pred_scaled = slope_size * x_scaled + intercept_size
    y_size_predicted = np.exp((y_size_pred_scaled * y_size_std) + y_size_mean)

    print("\nYear: {}\nCount: {}\nSize: {}".format(x, y_count_predicted, y_size_predicted))

if __name__ == '__main__':

    transistors_data = np.genfromtxt("transistor.csv", delimiter = ",")
    x = []
    y1 = []
    y2 = []
    for i in range(transistors_data.shape[0]):
        x.append(transistors_data[i][0])
        y1.append(transistors_data[i][1])
        y2.append(transistors_data[i][2])

    x_year = np.array(x)
    count = np.array(y1)
    size = np.array(y2)

    x_year_mean = np.mean(x_year)
    x_year_std = np.std(x_year)
    x_year_scaled = (x_year - x_year_mean)/x_year_std

    y_count = np.log(count)
    y_count_mean = np.mean(y_count)
    y_count_std = np.std(y_count)
    y_count_scaled = (y_count - y_count_mean)/y_count_std

    y_size = np.log(size)
    y_size_mean = np.mean(y_size)
    y_size_std = np.mean(y_size)
    y_size_scaled = (y_size - y_size_mean)/y_size_std

    learning_rate_count = 0.012
    learning_rate_size = 0.012

    plotRegression(x_year_scaled, y_count_scaled, learning_rate_count)
    plotRegression(x_year_scaled, y_size_scaled, learning_rate_size)

    predict(1987)
    print("\n")
