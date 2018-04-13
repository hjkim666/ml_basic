from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data 
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("train score: {:.2f}".format(lr.score(X_train, y_train)))
print("test score: {:.2f}".format(lr.score(X_test, y_test)))

import matplotlib.pyplot as plt 
plt.boxplot(X)
plt.show()


