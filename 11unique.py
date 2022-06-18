import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

df2 = df.nunique()
print(df2.mean())
