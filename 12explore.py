import pandas as pd

x_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/x_train.csv', encoding='cp949')
x_test = pd.read_csv('D:/python_workspace/gisa/bigData-main/x_test.csv', encoding='cp949')
y_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/y_train.csv', encoding='cp949')

#불필요한 컬럼 삭제