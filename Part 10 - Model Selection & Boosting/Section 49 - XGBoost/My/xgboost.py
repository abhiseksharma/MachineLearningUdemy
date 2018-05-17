# XGBoost

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataSet
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Label codeing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
x[:, 1] = labelencoder1.fit_transform(x[:, 1])
x[:, 2] = labelencoder2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Spliting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting XGBoost to Traning Set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)