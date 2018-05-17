# Artificial Neural Network

# Part 1 Data Preprocessing

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

# feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 Creating mdoel
# Import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising ANN
classifier = Sequential()

# Adding First layer which is a hidden layer defining number of inputs
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding second hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding output layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compilling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to train on training set 
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

# Part 3
# Prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred > .5)

# Comfusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)