from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
logreg = LogisticRegression()

kfold = KFold(n_splits=5)
#kfold = KFold(n_splits=3, shuffle=True, random_state=0)

print("accuracy:\n{}".format(
      cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
