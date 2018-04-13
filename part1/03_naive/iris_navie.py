from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
iris = load_iris()
print(iris)
X_iris = iris.data
y_iris = iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB() 
model.fit(Xtrain, ytrain)
predict = model.predict(Xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, predict)) 
