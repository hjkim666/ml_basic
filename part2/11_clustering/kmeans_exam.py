from sklearn.datasets import load_digits 
from sklearn.cluster import KMeans
import matplotlib.pylab as plt 

digits = load_digits() 
print(digits.data.shape)

kmeans = KMeans(n_clusters=10, random_state=10)
cluster = kmeans.fit_predict(digits.data)
# print(kmeans.cluster_centers_.shape)
# 
fig, ax = plt.subplots(2, 5, figsize=(8,3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
 
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()

