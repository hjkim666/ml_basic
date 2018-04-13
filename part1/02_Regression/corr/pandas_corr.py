import pandas as pd 

df = pd.DataFrame([[3000, 2000,3200,2100],
                [2000, 1700, 2600, 1800],
                [1500, 2000, 1500, 1300]])
print(df)
df = df.T

df.columns = ['a','b','c']

print(df)
print(df.corr())      