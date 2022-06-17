import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

#두줄코딩
df = df.sort_values(by='MEDV', ascending=True)
print(df['MEDV'].head(10))

#한줄코딩
print(df.sort_values(by='MEDV', ascending=True)['MEDV'].head(10))