from mglearn.datasets import make_blobs
from sklearn.model_selection  import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
import matplotlib.pylab as plt
import numpy as np

X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2],
                  random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svc = SVC(gamma=.05).fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# 0 근처의 임계값
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="threshold 0", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.show()
