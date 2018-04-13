
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

digits = load_digits()
y = digits.target == 8
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

tree = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
pred_tree = tree.predict(X_test)

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred_most_frequent,
                            target_names=["not 8", "8"]))



