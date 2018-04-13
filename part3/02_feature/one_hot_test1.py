import pandas as pd
import os
data = pd.read_csv("adult.data", header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
print(data.head())

print(data.gender.value_counts())
print("original feature:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("get_dummies:\n", list(data_dummies.columns))

print(data_dummies.head())

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
print(features)