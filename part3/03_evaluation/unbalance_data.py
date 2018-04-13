from sklearn.datasets import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.dummy import DummyClassifier
import numpy as np 

digits = load_digits()
y = digits.target == 8

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

#다수의 클래스를 예측값으로 내놓는 분류기 
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("most frequent class: {}".format(np.unique(pred_most_frequent)))
print("test accuracy: {:.2f}".format(dummy_majority.score(X_test, y_test)))



