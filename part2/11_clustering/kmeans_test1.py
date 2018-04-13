from sklearn.datasets.samples_generator import make_blobs 
import matplotlib.pylab as plt 

X, Y = make_blobs(n_samples = 300, centers=4
                      , cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1], s=50)
plt.show()

