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
        # print(slope)
        slope -= learning_rate * derivate_by_slope
        intercept -= learning_rate * derivate_by_intercept

    return (slope,intercept,loss)

def plotRegression(x, y, learning_rate):
    model = minimizeLossFunc(x, y, learning_rate)
    slope = model[0]
    intercept = model[1]
    loss = model[2]
    y_predicted = slope * x + intercept

    plt.scatter(x, y)
    plt.plot(x,y_predicted, color = 'red')
    plt.show()

def predict(x):
    x_scaled = (x - x_year_mean)/x_year_std

    model_count = minimizeLossFunc(x_year_scaled, y_count_scaled, learning_rate_count)
    slope_count = model_count[0]
    intercept_count = model_count[1]

    y_count_pred_scaled = slope_count * x_scaled + intercept_count
    y_count_predicted = np.exp((y_count_pred_scaled * y_count_std) + y_count_mean)

    print("Model Count in Predict: {}".format(model_count))
    # print("x_scaled: {}".format(x_scaled))
    # print("y_count_std: {}".format(y_count_std))
    # print("y_count_mean: {}".format(y_count_mean))
    # print("y_count_pred_scaled: {}".format(y_count_pred_scaled))
    # print("y_count_predicted: {}".format(y_count_predicted))

    model_size = minimizeLossFunc(x_year_scaled, y_size_scaled, learning_rate_size)
    slope_size = model_size[0]
    intercept_size = model_size[1]

    y_size_pred_scaled = slope_size * x_scaled + intercept_size
    y_size_predicted = np.exp((y_size_pred_scaled * y_size_std) + y_size_mean)

    print("Year: {}\nCount: {}\nSize: {}".format(x, y_count_predicted, y_size_predicted))

if __name__ == '__main__':

    transistors_data = pd.read_csv("transistor.csv", header = None)

    x_year = np.array(transistors_data[0])
    x_year_mean = np.mean(x_year)
    x_year_std = np.std(x_year)
    x_year_scaled = (x_year - x_year_mean)/x_year_std
    # print("x_year_scaled: {}".format(x_year_scaled))

    count = np.array(transistors_data[1])
    y_count = np.log(count)
    # print("y_count: {}".format(y_count))
    y_count_mean = np.mean(y_count)
    y_count_std = np.std(y_count)
    y_count_scaled = (y_count - y_count_mean)/y_count_std
    # print("y_count_scaled: {}".format(y_count_scaled))

    size = np.array(transistors_data[2])
    y_size = np.log(size)
    y_size_mean = np.mean(y_size)
    y_size_std = np.mean(y_size)
    y_size_scaled = (y_size - y_size_mean)/y_size_std
    # print("y_size_scaled: {}".format(y_size_scaled))

    # plotData(x_year, count, size)
    # plotData(x_year, y_count, y_size)

    learning_rate_count = optimizeLR(x_year_scaled, y_count_scaled)
    learning_rate_size = optimizeLR(x_year_scaled, y_size_scaled)
    plotRegression(x_year_scaled, y_count_scaled, learning_rate_count)
    plotRegression(x_year_scaled, y_size_scaled, learning_rate_size)

    predict(2020)
