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
line_binned = encoder.transform(np.digitize(line, bins=bins))
print(X_binned.shape)
X_combined = np.hstack([X, X_binned])
print(X_combined[0])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
print(line_combined[0])
plt.plot(line, reg.predict(line_combined)
         , label='regression added original feature')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
plt.legend(loc="best")
plt.ylabel("regression output")
plt.xlabel("feature")
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()

