from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

digits = load_digits()
y = digits.target == 8

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("DecisionTree accuracy: {:.2f}".format(tree.score(X_test, y_test)))

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy accuracy: {:.2f}".format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg accuracy: {:.2f}".format(logreg.score(X_test, y_test)))
