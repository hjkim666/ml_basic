from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

dataset = pd.read_csv("diabetes.csv",names=[1,2,3,4,5,6,7,8,9], header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum())  
print(dataset.describe())
print(dataset[9].head())

X = dataset[[1,2,3,4,5,6,7,8]]
y = dataset[9]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size = .7, random_state=107)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
 
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
 
print(logisticRegr.predict(X_test[0:10]))   
predictions = logisticRegr.predict(X_test)

 
print("logistic:{}.".format(logisticRegr.score(X_test, y_test)))

##############################################
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
print("KNN:{}".format(clf.score(X_test,y_test)))
##############################################
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X_train, y_train)
print("Tree:{}".format(tree.score(X_test, y_test)))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=50, random_state=0)
forest.fit(X_train, y_train)
print("forest:{}:".format(forest.score(X_test,y_test)))
##############################################
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("SVM:{}".format(svc.score(X_test, y_test)))
##############################################
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=2,random_state=42,max_iter=10000)
mlp.fit(X_train, y_train)
print("MLP:{}".format(mlp.score(X_test, y_test)))

  
