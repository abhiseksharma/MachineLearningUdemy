# Hierarchical Clustering

#%reset -f

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:5].values

# Ploting Dendrogram
import scipy.cluster.hierarchy as scp
dengrogram = scp.dendrogram(scp.linkage(x, method = 'ward'))
plt.show()

# Fitting Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

# Plotting
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'first')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'green', label = 'second')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'blue', label = 'third')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'forth')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'yellow', label = 'fifth')
plt.legend()
plt.show()