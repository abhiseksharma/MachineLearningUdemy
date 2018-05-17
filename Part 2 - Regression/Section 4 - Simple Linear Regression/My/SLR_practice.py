# SLR practice

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = .2)

# regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predict result
ypred = regressor.predict(xtest)

# Ploting
plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtest, regressor.predict(xtest), color = 'blue')
plt.show()