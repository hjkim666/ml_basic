#-*- coding: utf-8 -*-
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs 
import numpy as np 
import matplotlib.pylab as plt

def find_clusters(X, n_clusters, rseed=2):
    #1.임의의 군집 선택 
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        #2a. 가장 가까운 중심을 기반으로 레이블 할당 
        labels = pairwise_distances_argmin(X, centers)
        #2b. 점들의 평균으로부터 새로운 군집 발견  
        new_centers = np.array([X[labels==i].mean(0) for i in range(n_clusters)])
        #2c.수렴여부 검사 
        if np.all(centers == new_centers):
            break
        centers = new_centers 
    return centers, labels 

X, Y = make_blobs(n_samples = 300, centers=4
                      , cluster_std=0.60, random_state=0)    
centers , labels = find_clusters(X, 4)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')
plt.show()    
        
# centers , labels = find_clusters(X, 4, rseed=0)
# plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')
# plt.show()        
            
            

