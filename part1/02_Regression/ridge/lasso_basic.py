from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import numpy as np 

def load_extended_boston():
    boston = load_boston()
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target

X, y = load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("train accuracy: {:.2f}".format(lasso.score(X_train, y_train)))
print("test accuracy: {:.2f}".format(lasso.score(X_test, y_test)))
print("feature count: {:.2f}".format(np.sum(lasso.coef_ != 0)))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
asso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("train accuracy: {:.2f}".format(lasso001.score(X_train, y_train)))
print("test accuracy: {:.2f}".format(lasso001.score(X_test, y_test)))
print("feature count: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("train accuracy: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("test accuracy: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("feature count: {}".format(np.sum(lasso00001.coef_ != 0)))
