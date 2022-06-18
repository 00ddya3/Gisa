import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

df = df['MEDV'].sort_values(ascending=False)
a = df.iloc[29]
print(a)

df.iloc[0:29] = a
print(df.mean(), df.median(), df.min(), df.max())