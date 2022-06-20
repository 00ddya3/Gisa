import pandas as pd
import warnings
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings(action='ignore')


# 데이터 불러오기
x_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/bike_x_train.csv', encoding='cp949')
x_test = pd.read_csv('D:/python_workspace/gisa/bigData-main/bike_x_test.csv', encoding='cp949')
y_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/bike_y_train.csv')

# 전처리를 위해 train, test 합체
x_train['label'] = 'train'
x_test['label'] = 'test'
x_data = pd.concat([x_train, x_test], axis=0)

# 다중공선성 제거
#print(x_data.corr())
x_data.drop(columns='온도', inplace=True)

# datetime
x_data['datetime'] = pd.to_datetime(x_data['datetime'])
x_data['year'] = x_data['datetime'].dt.year
x_data['month'] = x_data['datetime'].dt.month
x_data['hour'] = x_data['datetime'].dt.hour

#필요없는 열 제거
x_test_datetime = x_test['datetime']
x_data.drop(columns=['datetime'], inplace=True)
y_train.drop(columns=['datetime'], inplace=True)

# 다시 나눠주기
x_train = x_data[x_data['label'] == 'train']
x_test = x_data[x_data['label'] == 'test']
x_train.drop(columns=['label'], inplace=True)
x_test.drop(columns=['label'], inplace=True)


# train data를 train, validation으로 나누기
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train)

# 모델링
model = XGBRegressor(n_estimators=100, max_depth=3, random_state=10)        #0.8981
#model = XGBRegressor(n_estimators=200, max_depth=5, random_state=10)        #0.9421
model.fit(X_train, Y_train)
Y_predict = model.predict(X_val)
Y_predict = pd.DataFrame(Y_predict, columns=['count'])
Y_predict[Y_predict['count'] < 0] = 0       #count값이기에 음수는 0으로 처리

#모델 평가하기
print(r2_score(Y_predict, Y_val))


# 실제 test data로 예측하기
y_predict = model.predict(x_test)
y_predict = pd.DataFrame(y_predict, columns=['count'])
y_predict[y_predict['count'] < 0] = 0       #count값이기에 음수는 0으로 처리

# 결과 제출하기
finaldf = pd.concat([x_test_datetime, y_predict], axis=1)
finaldf.to_csv('D:/python_workspace/gisa/bigData-main/bike_final.csv', index=False)

#print(x_train.head())