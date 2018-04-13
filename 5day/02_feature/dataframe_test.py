import pandas as pd 

demo_df = pd.DataFrame({'numeric feature':[0,1,2,1],
                       'categorical feature':['book', 'pen','book','box']} )
print(demo_df)

print(pd.get_dummies(demo_df))


demo_df['numeric feature'] = demo_df['numeric feature'].astype(str)
print(pd.get_dummies(demo_df, columns=['numeric feature','categorical feature'])) 

