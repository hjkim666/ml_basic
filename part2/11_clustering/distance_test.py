from sklearn.neighbors import DistanceMetric

dist = DistanceMetric.get_metric('euclidean')
X = [[0, 1, 2],
    [3, 4, 5]]
d = dist.pairwise(X)
print(d)

X2 = [[1,2,3,4,5,6,7,8,9,10],
      [11,12,13,14,15,16,17,18,19,10],
      [21,22,23,24,25,26,27,28,29,210]]
d = dist.pairwise(X2)
print(d)