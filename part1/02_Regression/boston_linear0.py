import pandas as pd
from sklearn import linear_model
from sklearn.datasets import load_boston  
data = load_boston()  

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
print(predictions[0:5])
print("score:{}".format(model.score(X, y)))

