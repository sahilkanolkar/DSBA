import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def plotData(x, y1, y2):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x, y1)
    ax[1].scatter(x, y2)
    plt.show()

def optimizeLR(x, y):
    """
    Test values of Learning Rate between 0.0000001 to 100
    """

    learning_rate = 0.0000001
    optimized_LR = learning_rate
    min_loss_function = 100

    while (learning_rate <= 100):
        loss = minimizeLossFunc(x, y, learning_rate)[2][-1]
        if loss < min_loss_function:
            min_loss_function = loss
            optimized_LR = learning_rate

        learning_rate *= 2
    return optimized_LR

def minimizeLossFunc(x, y, learning_rate):
    n = len(y)
    slope = 0
    intercept = 0
    loss = []

    for i in range(n):
        y_predicted = slope * x + intercept
        loss.append((1/n) * sum((y - y_predicted) ** 2))
        derivate_by_slope = (-2/n) * sum(x * (y - y_predicted))
        derivate_by_intercept = (-2/n) * sum(y - y_predicted)
        slope -= learning_rate * derivate_by_slope
        intercept -= learning_rate * derivate_by_intercept

    return (slope,intercept,loss)

# def predict(x):
#     model_count = minimizeLossFunc(y_count_scaled)
#     slope_count = model_count[0]
#     # print("Model Count in Predict: {}".format(model_count))
#     intercept_count = model_count[1]
#     x_scaled = (x - x_year_mean)/x_year_std
#     y_pred_scaled = slope_count * x_scaled + intercept_count
#     y_count_predicted = np.exp(((slope_count * x_scaled + intercept_count) * y_count_std) + y_count_mean)
#     # print("x_scaled: {}".format(x_scaled))
#     # print("y_count_std: {}".format(y_count_std))
#     # print("y_count_mean: {}".format(y_count_mean))
#     # print("y_pred_scaled: {}".format(y_pred_scaled))
#
#     model_size = minimizeLossFunc(y_size)
#     slope_size = model_size[0]
#     intercept_size = np.exp(model_size[1])
#     y_size_predicted = np.exp(slope_size * x + intercept_size)
#
#     print("Year: {}\nCount: {}\nSize: {}".format(x, y_count_predicted, y_size_predicted))

def plotRegression(x, y, learning_rate):
    model = minimizeLossFunc(x, y, learning_rate)
    slope = model[0]
    intercept = model[1]
    loss = model[2]
    y_predicted = slope * x + intercept

    plt.scatter(x, y)
    plt.plot([min(x), max(x)],[min(y_predicted), max(y_predicted)], color = 'red')
    plt.show()

if __name__ == '__main__':

    transistors_data = pd.read_csv("transistor.csv", header = None)

    x_year = np.array(transistors_data[0])
    x_year_mean = np.mean(x_year)
    x_year_std = np.std(x_year)
    x_year_scaled = (x_year - x_year_mean)/x_year_std

    count = np.array(transistors_data[1])
    y_count = np.log(count)
    y_count_mean = np.mean(y_count)
    y_count_std = np.std(y_count)
    y_count_scaled = (y_count - y_count_mean)/y_count_std

    size = np.array(transistors_data[2])
    y_size = np.log(size)
    y_size_mean = np.mean(y_size)
    y_size_std = np.mean(y_size)
    y_size_scaled = (y_size - y_size_mean)/y_size_std

    # plotData(x_year, count, size)
    # plotData(x_year, y_count, y_size)

    learning_rate_count = optimizeLR(x_year_scaled, y_count_scaled)
    learning_rate_size = optimizeLR(x_year_scaled, y_size_scaled)
    plotRegression(x_year_scaled, y_count_scaled, learning_rate_count)
    plotRegression(x_year_scaled, y_size_scaled, learning_rate_size)
