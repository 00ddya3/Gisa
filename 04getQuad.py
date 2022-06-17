import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

df.drop(['CHAS', 'RAD'], axis=1, inplace=True)      #필요없는 열 제거

