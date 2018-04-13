#-*- coding: utf-8 -*-
from sklearn.datasets import make_moons 
from sklearn.cluster import SpectralClustering
from matplotlib.pylab import plt

x, y = make_moons(200, noise=.05, random_state=0)

model = SpectralClustering(n_clusters=2
                           , affinity='nearest_neighbors'
                           , assign_labels='kmeans')
labels = model.fit_predict(x)
plt.scatter(x[:,0], x[:,1], c=labels, s=50, cmap='viridis')
plt.show()


