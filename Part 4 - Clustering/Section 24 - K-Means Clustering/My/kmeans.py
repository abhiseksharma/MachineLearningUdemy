# Kmeans

# importing Lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Imprting dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:5].values

# Using elbow method to find number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.show()

# Applying Kmeans to mall dataset
kmeans = KMeans(n_clusters = 5, n_init = 10, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Ploting
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Target')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.legend()
plt.show()