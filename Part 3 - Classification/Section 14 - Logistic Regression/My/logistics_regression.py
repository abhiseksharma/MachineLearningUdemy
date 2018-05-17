# Logistics Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Splitting Dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
Xtrain = scx.fit_transform(Xtrain)
Xtest = scx.fit_transform(Xtest)

# Fitting Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xtrain, ytrain)

# Testing Model by predicting values for xtest
ypred = classifier.predict(Xtest)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

# Visualizing the training set results
