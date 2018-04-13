# t-sne
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode 
import matplotlib.pylab as plt 
import numpy as np 


digits = load_digits() 

tsne = TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)
print(kmeans.cluster_centers_.shape)

labels = np.zeros_like(clusters)
for i in range(10):
    mask=(clusters==i)
    labels[mask] = mode(digits.target[mask])[0]
    
print(accuracy_score(digits.target, labels))   