import pandas as pd
from scipy.stats import mode

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

#방법1
df['AGE'] = df['AGE'].round(0)      #반올림
df1 = df['AGE'].value_counts()      #개수세기
print(df1.head(1))

#방법2
df2 = df['AGE'].round(0)      #반올림
print(int(mode(df2)[0]), int(mode(df2)[1]))     #mode: 최빈값계산