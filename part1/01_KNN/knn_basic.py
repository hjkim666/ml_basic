from sklearn.datasets import make_blobs
X, y = make_blobs(centers=2, random_state=4, n_samples=30)
######################################################
import mglearn 
import matplotlib.pylab as plt
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["class 0", "class 1"], loc=4)
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.show()
print("X.shape: {}".format(X.shape))
#######################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#######################################################
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=3, p=2,
#            weights='uniform')
print("test prediction: {}".format(clf.predict(X_test)))
print("test accuracy: {:.2f}".format(clf.score(X_test, y_test)))
