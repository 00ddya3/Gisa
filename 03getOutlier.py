import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

meanZN = df.ZN.mean()
stdZN = df.ZN.std()

upper = meanZN + 1.5*stdZN
lower = meanZN - 1.5*stdZN

print(df[(df['ZN'] > upper) | (df['ZN'] < lower)].ZN.sum())