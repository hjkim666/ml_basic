from sklearn.datasets import load_boston
import mglearn

boston = load_boston()
print("shape: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

#############################################
