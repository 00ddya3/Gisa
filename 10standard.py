import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

scaler = StandardScaler()
df_stdd = scaler.fit_transform(df)

df2 = pd.DataFrame(df_stdd, columns=df.columns)

df3 = df2[(df2['DIS'] > 0.4) & (df2['DIS'] < 0.6)]['DIS']
print(round(df3.mean(),2))
