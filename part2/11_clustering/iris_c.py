from sklearn.cluster import KMeans
import matplotlib.pylab as plt 
import numpy as np 

xy = np.loadtxt('iris_training.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

kmeans = KMeans(n_clusters=3, random_state=10)
cluster = kmeans.fit_predict(x_data)
print(kmeans.cluster_centers_.shape)
 
plt.scatter(x_data[:,0], x_data[:,1], c=cluster, s=50, cmap='viridis')
#plt.scatter(x_data[:,1], x_data[:,2], c=cluster, s=50, cmap='viridis')
#plt.scatter(x_data[:,2], x_data[:,3], c=cluster, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.show()

