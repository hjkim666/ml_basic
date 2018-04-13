from sklearn.neighbors import DistanceMetric

dist = DistanceMetric.get_metric('euclidean')
X = [[0, 1, 2],
     [3, 4, 5]]
d = dist.pairwise(X)
print(d)