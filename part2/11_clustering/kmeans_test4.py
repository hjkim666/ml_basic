#-*- coding: utf-8 -*-
from sklearn.datasets import make_moons 
from sklearn.cluster import KMeans
from matplotlib.pylab import plt

x, y = make_moons(200, noise=.05, random_state=0)
print(x,y)

labels = KMeans(2, random_state=0).fit_predict(x)
plt.scatter(x[:,0], x[:,1], c=labels, s=50, cmap='viridis')
plt.show()

