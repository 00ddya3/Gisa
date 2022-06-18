import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

scaler = MinMaxScaler()

minmax_df = scaler.fit_transform(df)        #ndarray type
minmax_df = pd.DataFrame(minmax_df, columns=df.columns)      #dataframe type

#print(minmax_df.MEDV.describe())
print(minmax_df[minmax_df['MEDV'] > 0.5]['MEDV'].count())