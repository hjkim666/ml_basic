from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
import mglearn
import matplotlib.pylab as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)

encoder = OneHotEncoder(sparse=False)
which_bin = np.digitize(X, bins=bins)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned)
line_binned = encoder.transform(np.digitize(line, bins=bins))

X_product = np.hstack([X_binned, X*X_binned])
print(X_product[0])

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
print(line_product[0])
plt.plot(line, reg.predict(line_product)
         , label='regression added original feature')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
plt.legend(loc="best")
plt.ylabel("regression output")
plt.xlabel("feature")
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()

