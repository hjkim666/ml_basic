import numpy as np 

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    # 클래스마다 반복하며, 특성마다 1이 나타난 횟수를 카운트 
    counts[label] = X[y == label].sum(axis=0)
print("feature count:\n{}".format(counts))

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(X, y)
print(model.score(X,y))