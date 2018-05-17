# Random Forest Regresion

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
X = pd.read_csv('Position_Salaries.csv').iloc[:, 1:-1].values

y = pd.read_csv('Position_Salaries.csv').iloc[:, -1].values
#X = dataset.iloc[:, 1:-1].values
#y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
'''

# Creating Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

plt.scatter(xp[y_hc == 0, 0], xp[y_hc == 0, 1], s = 100, c = 'red', label = 'first')
plt.scatter(xp[y_hc == 1, 0], xp[y_hc == 1, 1], s = 100, c = 'green', label = 'second')
'''plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'blue', label = 'third')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'forth')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'yellow', label = 'fifth')'''
plt.legend()
plt.show()

# Prediction
y_pred = regressor.predict(6.5)

# Ploting
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()