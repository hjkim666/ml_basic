from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Train Accuracy: {:.3f}".format(forest.score(X_train, y_train)))
print("Test Accuracy: {:.3f}".format(forest.score(X_test, y_test)))

########################################################################
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# defult depth:3, tree:100개, learning rate=0.1
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("train accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("test accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("train accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("test accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("train accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("test accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
####################################################################
print("feature importance:\n{}".format(gbrt.feature_importances_))

import matplotlib.pylab as plt
import numpy as np 

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.show() 
    
plot_feature_importances_cancer(gbrt)
#mglearn.plot_feature_importances_cancer(gbrt)