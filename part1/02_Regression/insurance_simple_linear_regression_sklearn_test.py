import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#plt.ion()

## Data Collection, Preprocessing, Exploration

dataset = pd.read_csv("insurance.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum()) # check missing values
print(dataset.describe())

# print top 6 the charges variable
print(dataset['expenses'].head())

# histogram of insurance charges
plt.hist(dataset['expenses'], bins=10)
plt.show()

# visualing relationships between expendes, ages
#plt.figure()
plt.plot(dataset['age'], dataset['expenses'], 'o')

# Split data into training and test data
X = dataset[['age']]
y = dataset['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state=107)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## Training a linear regression model

model = LinearRegression()
model.fit(X_train, y_train)
print(model)
print(model.coef_)
print(model.intercept_)
print("score:{}".format(model.score(X_train, y_train)))

x = np.array([min(dataset['age']), max(dataset['age'])]).reshape(-1,1)
print(model.predict(x))
plt.plot(x, model.predict(x))
plt.show()
## Performance Evaluation

# training error
model.score(X_train, y_train)
pred = model.predict(X_train)
pred[0:6]
plt.plot(X_train['age'], pred, c='red')
np.sqrt(mean_squared_error(y_train, pred))
np.sqrt(sum((pred - y_train)**2)/(y_train.shape[0]-2))  # residual standard error

# predictions on some data
pred = model.predict(pd.DataFrame({'age': [20, 40, 60]}))
print(pred)

# predict for test data
model.score(X_test, y_test)
pred = model.predict(X_test)
str(pred)
np.sqrt(mean_squared_error(y_test, pred))
