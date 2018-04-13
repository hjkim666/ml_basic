import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

plt.ion()

## Data Collection, Preprocessing, Exploration

dataset = pd.read_csv("insurance.csv")
dataset.head()
dataset.shape
dataset.isnull().sum().sum() # check missing values
dataset.describe()
dataset['expenses'].head()

# histogram of insurance charges
dataset['region'].value_counts()
plt.hist(dataset['expenses'], bins=10)
plt.show()

# correlation analysis, scatter plot
np.corrcoef([dataset["age"], dataset["bmi"], dataset["children"], dataset["expenses"]])
plt.figure()
pd.tools.plotting.scatter_matrix(dataset[["age", "bmi", "children", "expenses"]])

# convert categorical data to one hot encoding
#dataset = pd.get_dummies(dataset, columns=["sex", "smoker", "region"], drop_first=True)
dataset["sex"].value_counts()
dataset["smoker"].value_counts()
dataset["region"].value_counts()

def binarize(df):
    from sklearn.preprocessing import label_binarize
    sex_male = pd.DataFrame(label_binarize(df["sex"], classes=["female", "male"]), columns=["sex_male"])
    smoker_yes = pd.DataFrame(label_binarize(df["smoker"], classes=["no", "yes"]), columns=["smoker_yes"])
    region = pd.DataFrame(label_binarize(df["region"], classes=["northeast", "northwest", "southeast", "southwest"])[:,1:], 
                      columns=["region_northwest", "region_southeast", "region_southwest"])
    return pd.concat([sex_male, smoker_yes, region], axis=1)

# Split data into training and test data
X = pd.concat([dataset[["age", "bmi", "children"]], binarize(dataset)], axis=1)
y = dataset["expenses"]
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, train_size = .7, random_state=107)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## Training a linear regression model

model = sk.linear_model.LinearRegression()
model.fit(X_train, y_train)
model.coef_
model.intercept_

## Performance Evaluation

# training error
model.score(X_train, y_train)
pred = model.predict(X_train)
np.sqrt(sk.metrics.mean_squared_error(y_train, pred))
pred[0:6]

# predict for some data
someData = pd.DataFrame({"age": [46], "sex": ["female"], "bmi": [33], "children": [1], "smoker": ["no"], "region": ["southeast"]})
someData2 = pd.concat([someData[["age", "bmi", "children"]], binarize(someData)], axis=1)
pred = model.predict(someData2)
pred

# predict for test data
model.score(X_test, y_test)
pred = model.predict(X_test)
np.sqrt(sk.metrics.mean_squared_error(y_test, pred))

## Performance Enhancement (higher-order "age" term, BMI >= 30 indicator and smoker)

# add a higher-order "age" term
X_train.loc[:,"age2"] = X_train["age"]**2
X_test.loc[:,"age2"] = X_test["age"]**2

# add an indicator for BMI >= 30
plt.figure()
plt.plot(dataset["bmi"], dataset["expenses"], 'o')
X_train.loc[:,"bmi30"] = np.where(X_train["bmi"] >= 30, 1, 0)
X_test.loc[:,"bmi30"] = np.where(X_test["bmi"] >= 30, 1, 0)

# add an composite indicator
X_train.loc[:,"bmi30_smoker"] = X_train["bmi30"] * X_train["smoker_yes"]
X_test.loc[:,"bmi30_smoker"] = X_test["bmi30"] * X_test["smoker_yes"]
X_test2 = X_test[["age", "age2", "children", "bmi", "sex_male", "bmi30", "smoker_yes", "bmi30_smoker", "region_northwest", "region_southeast", "region_southwest"]]

model2 = sk.linear_model.LinearRegression()
model2.fit(X_train2, y_train)
model2.coef_
model2.intercept_

model2.score(X_test2, y_test)
pred = model2.predict(X_test2)
np.sqrt(sk.metrics.mean_squared_error(y_test, pred))
