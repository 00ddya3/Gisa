import pandas as pd

x_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/titanic_x_train.csv', encoding='cp949')

print(x_train['나이'].median())