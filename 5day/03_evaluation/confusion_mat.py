from sklearn.datasets import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits = load_digits()
y = digits.target == 8
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

confusion = confusion_matrix(y_test, pred_logreg)
print("confusion matrix:\n{}".format(confusion))

#알고리즘별 비교 
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

print("frequent dummy model:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nrandom dummy model:")
print(confusion_matrix(y_test, pred_dummy))
print("\ndecision tree:")
print(confusion_matrix(y_test, pred_tree))
print("\nlogistic regression:")
print(confusion_matrix(y_test, pred_logreg))

