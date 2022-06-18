import pandas as pd

df = pd.read_csv('D:/python_workspace/gisa/bigData-main/boston.csv')

medTAX = df['TAX'].median()

df = df[df['TAX'] > medTAX]

df = df.groupby(['CHAS', 'RAD'])['RAD'].count()
df2 = pd.DataFrame(df)      #이거 안해주면 컬럼명 지정 못함!!!
df2.columns = ['COUNT']
print(df2)