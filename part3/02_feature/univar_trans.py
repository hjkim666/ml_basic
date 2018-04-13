import numpy as np 
import matplotlib.pylab as plt 

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
print(X[:10, 0])

print("feature count:\n{}".format(
    np.bincount(X[:, 0].astype('int'))))

plt.xlim(0, 160)
plt.ylim(0, 70)
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='grey')
plt.ylabel("count") #출현회수 
plt.xlabel("value") #값
plt.show()

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("test score: {:.3f}".format(score))

#log scale
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.hist(X_train_log[:,0], bins=25, color='gray')
plt.ylabel("count")
plt.xlabel("value")
plt.show()

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("test score:{:.3f}".format(score))





