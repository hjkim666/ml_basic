from sklearn.preprocessing import OneHotEncoder
import numpy as np 
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=100)
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

print("X_binned.shape:{}".format(X_binned.shape))

