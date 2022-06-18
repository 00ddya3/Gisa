import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

df1 = df['TAX'].sort_values(ascending=True).reset_index(drop=True)      #현재 인덱스 정보를 남기지 않겠다
df2 = df['TAX'].sort_values(ascending=False).reset_index(drop=True)     #현재 인덱스 정보를 남기지 않겠다

df3 = abs(df1 - df2)        #문제에 절댓값으로 계산하라는 말은 없었는데 눈치껏 해야하는 듯?
print(df3['TAX'].var())