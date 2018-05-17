# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

'''
# Taking care of mising data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_C = LabelEncoder()
X[:, 0] = labelencoder_C.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_x = LabelEncoder()
y = labelencoder_x.fit_transform(y)
'''

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

# Feature Scaling
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
from sklearn.metrics import r2_score
print(round(r2_score(y_test, y_pred)*100, 2) , 'percent')