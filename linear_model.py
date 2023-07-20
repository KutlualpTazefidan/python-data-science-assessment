# Alp
# Fit the imported data using a linear model

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.system("clear")

# Task 1: Import Data Points
dataPointsPath = "./DataPoints.csv"
dataPointsFrame = pd.read_csv(dataPointsPath)
xTrueValues = dataPointsFrame["x"]
yTrueValues = dataPointsFrame["y"]

# Task 3: Linear equation
slope = 10
intercept = 0
yValues = slope * xTrueValues + intercept

# Task 4: MSE - Mean Squared Error
mseInitial = np.mean((yValues - yTrueValues) ** 2)

# Task 5: Slope Optimization
mseOpt = mseInitial
slopeMSE = 10
interceptMSE = 0
for i in range(100):
    yValues = slope * xTrueValues + interceptMSE
    mseUpdated = np.mean((yValues - yTrueValues) ** 2)
    if mseUpdated < mseOpt:
        mseOpt = mseUpdated
        slopeMSE = slope
    slope -= 0.1
yMSE = slopeMSE * xTrueValues + interceptMSE

# Task 6: Intercept Optimization - I don't get the task
mseOpt2 = mseInitial
slope = 10
intercept = 0
slopeTask6 = 10
interceptTask6 = 0
for i in range(100):
    yValues = slope * xTrueValues + intercept
    mseUpdated = np.mean((yValues - yTrueValues) ** 2)
    if mseUpdated < mseOpt2:
        mseOpt2 = mseUpdated
        slopeTask6 = slope
        interceptTask6 = intercept
    slope -= 0.1
    intercept -= 0.1

# Task 7:
# a) Linear Regression
xTrueMean = np.mean(xTrueValues)
yTrueMean = np.mean(yTrueValues)
xDelta = xTrueValues - xTrueMean
yDelta = yTrueValues - yTrueMean
slopeRegression = sum(xDelta * yDelta) / sum(xDelta**2)
interceptRegression = yTrueMean - (slopeRegression * xTrueMean)
yValuesRegression = slopeRegression * xTrueValues + interceptRegression
mseRegression = np.mean((yValuesRegression - yTrueValues) ** 2)

# b) Gradient Descent
slopeGD = slopeMSE
interceptGD = 0.0
learningRate = 0.001
iterations = 10000

# derivative of the sum of squared residuals
# sum of the squared residuals: 1/len(x) * sum()

for i in range(iterations):
    # y estimation
    yGD = slopeGD * xTrueValues + interceptGD

    # gradients
    dSlope = -2 * np.mean(xTrueValues * (yTrueValues - yGD))
    dIntercept = -2 * np.mean(yTrueValues - yGD)

    slopeGD -= learningRate * dSlope
    interceptGD -= learningRate * dIntercept

yValuesGD = slopeGD * xTrueValues + interceptGD
mseGD = np.mean((yValuesGD - yTrueValues) ** 2)

# Print output
print(
    "Task 4: Slope:",
    10,
    ", Intercept:",
    0,
    ", MSE",
    mseInitial,
    "Comment: initial values",
)

print(
    "Task 5: Slope:",
    slopeMSE,
    ", Intercept:",
    interceptMSE,
    ", MSE:",
    mseOpt,
    ", Comment: Slope optimization using MSE",
)

print(
    "Task 6: Slope:",
    slopeTask6,
    ", Intercept:",
    interceptTask6,
    ", MSE:",
    mseOpt2,
    ", Comment: Instead use Linear Regression!",
)

print(
    "Task 7: a) Slope:",
    slopeRegression,
    ", Intercept:",
    interceptRegression,
    ", MSE:",
    mseRegression,
    ", Comment: Linear regression",
)

print(
    "Task 7: b) Slope:",
    slopeGD,
    ", Intercept:",
    interceptGD,
    ", MSE:",
    mseGD,
    ", Comment: Gradient Descent",
)

# Plot results Task 2
fontSize = 16
# Input
plt.grid()
plt.scatter(
    xTrueValues,
    yTrueValues,
    label="Data Points",
    color="black",
    marker="o",
)
plt.title("Results", fontsize=fontSize)
plt.xlabel("x Values", fontsize=fontSize)
plt.ylabel("y Values", fontsize=fontSize)

# Slope MSE
plt.plot(
    xTrueValues,
    yMSE,
    label="Slope MSE",
    color="blue",
)

# Linear Regression
plt.plot(
    xTrueValues,
    yValuesRegression,
    label="Linear Regression",
    color="red",
)

# Gradient Descent
plt.plot(
    xTrueValues,
    yValuesGD,
    label="Gradient Descent",
    color="green",
)
plt.legend(fontsize=fontSize)
plt.tick_params(axis="both", which="major", labelsize=fontSize)
plt.tight_layout()
plt.xlim(0, max(xTrueValues) + 0.1)
plt.ylim(0, max(yTrueValues) + 0.1)
plt.show()
# plt.gca().xaxis.set_label_coords(1.0, 0)
# plt.gca().yaxis.set_label_coords(0, 1.0)
# plt.draw()
