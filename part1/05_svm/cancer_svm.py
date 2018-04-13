from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pylab as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Train Accuracy: {:.2f}".format(svc.score(X_train, y_train)))
print("Test Accuracy: {:.2f}".format(svc.score(X_test, y_test)))

plt.boxplot(X_train, manage_xticks=False)
plt.yscale("symlog")
plt.xlabel("feature list")
plt.ylabel("feature size")
plt.show()

###########################################################
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면 각 특성에 대해 최솟값은 0 최댓값은 1 임
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
print("min:\n{}".format(X_train_scaled.min(axis=0)))
print("max:\n {}".format(X_train_scaled.max(axis=0)))

svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Train Accuracy: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Test Accuracy: {:.3f}".format(svc.score(X_test_scaled, y_test)))

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Train Accuracy: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Test Accuracy: {:.3f}".format(svc.score(X_test_scaled, y_test)))
