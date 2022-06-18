import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

df.drop(['CHAS', 'RAD'], axis=1, inplace=True)      #필요없는 열 제거

#방법1
columnlist = df.columns

iqrlist = []
for column in columnlist :
    iqrlist.append(df[column].quantile(0.75) - df[column].quantile(0.25))

df2 = pd.DataFrame(data=[columnlist, iqrlist])
df2 = df2.transpose()
print(df2)


# 방법2
desdf = df.describe()

df3 = desdf.iloc[6] - desdf.iloc[3]

df3 = df3.reset_index()
print(df3)