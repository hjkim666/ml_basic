import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

## Data Collection, Preprocessing, Exploration
dataset = pd.read_csv("test-score.csv",names=['1st','2nd','3rd','final'], header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum()) # check missing values
print(dataset.describe())

# print top 6 the charges variable
print(dataset['final'].head())

# # Split data into training and test data
X = dataset[['1st','2nd','3rd']]
y = dataset['final']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state=107)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ## Training a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
print(model)
print(model.coef_)
print(model.intercept_)
print("score:{}".format(model.score(X_train, y_train)))
