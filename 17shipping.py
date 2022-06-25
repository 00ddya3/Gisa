import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score



warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', 20)

#데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv")

#결측치 - 없음

#필요없는 데이터 제거
x_test_id = x_test['ID']
x_train.drop(columns = 'ID', inplace=True)
x_test.drop(columns = 'ID', inplace=True)
y_train.drop(columns = 'ID', inplace=True)

#Customer_care_calls 오타수정
x_train['Customer_care_calls'] = x_train['Customer_care_calls'].map(lambda x : x.replace('$', ''))
x_train['Customer_care_calls'] = x_train['Customer_care_calls'].astype('int')
x_test['Customer_care_calls'] = x_test['Customer_care_calls'].map(lambda x : x.replace('$', ''))
x_test['Customer_care_calls'] = x_test['Customer_care_calls'].astype('int')

#라벨인코딩 - Warehouse_block
encoder = LabelEncoder()
x_train['Warehouse_block'] = encoder.fit_transform(x_train['Warehouse_block'])
x_test['Warehouse_block'] = encoder.transform(x_test['Warehouse_block'])

#라벨인코딩 - Mode_of_Shipment
x_train['Mode_of_Shipment'] = encoder.fit_transform(x_train['Mode_of_Shipment'])
x_test['Mode_of_Shipment'] = encoder.transform(x_test['Mode_of_Shipment'])

#라벨인코딩 - Product_importance
x_train['Product_importance'] = encoder.fit_transform(x_train['Product_importance'])
x_test['Product_importance'] = encoder.transform(x_test['Product_importance'])

#라벨인코딩 - Gender  
x_train['Gender'] = x_train['Gender'].map(lambda x : 0 if x == 'F' else 1) 
x_test['Gender'] = x_test['Gender'].map(lambda x : 0 if x == 'F' else 1) 

#스케일링 - Cost_of_the_Product
scaler = MinMaxScaler()
x_train['Cost_of_the_Product'] = scaler.fit_transform(x_train[['Cost_of_the_Product']])
x_test['Cost_of_the_Product'] = scaler.transform(x_test[['Cost_of_the_Product']])

#스케일링- Discount_offered
x_train['Discount_offered'] = scaler.fit_transform(x_train[['Discount_offered']])
x_test['Discount_offered'] = scaler.transform(x_test[['Discount_offered']])

#스케일링- Weight_in_gms
x_train['Weight_in_gms'] = scaler.fit_transform(x_train[['Weight_in_gms']])
x_test['Weight_in_gms'] = scaler.transform(x_test[['Weight_in_gms']])


# 데이터 나누기
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)

#모델링
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_val)
Y_predict = pd.DataFrame(Y_predict, columns = ['Reached.on.Time_Y.N'])

#검증
print('랜포 : ', roc_auc_score(Y_val, Y_predict))

#결과제출
y_predict = model.predict(x_test)
y_predict = pd.DataFrame(y_predict, columns = ['Reached.on.Time_Y.N'])

df_final = pd.concat([x_test_id, y_predict], axis=1)
#df_final.to_csv('@@@.csv', index=False)

y_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_test.csv")
print('랜포 : ', roc_auc_score(y_predict, y_test['Reached.on.Time_Y.N']))