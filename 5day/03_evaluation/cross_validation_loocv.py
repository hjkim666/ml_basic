from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("cv spilt count: ", len(scores))  #교차 검증 분할 횟수
print("mean of accuracy: {:.2f}".format(scores.mean()))

