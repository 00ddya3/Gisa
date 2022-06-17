from numpy import average
import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

aveRM = df['RM'].fillna(df['RM'].mean())
rmvRM = df['RM'].dropna()

std1 = aveRM.std()
std2 = rmvRM.std()

print(abs(std1 - std2))