from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np 
import mglearn
import matplotlib.pylab as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
print(X_poly.shape)
print(X[:5])
print(X_poly[:5])
print(poly.get_feature_names())

reg = LinearRegression().fit(X_poly, y)

line_poly =  poly.transform(line)
plt.plot(line, reg.predict(line_poly)
         , label='polynomial regression')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
plt.legend(loc="best")
plt.ylabel("regression output")
plt.xlabel("feature")
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()

