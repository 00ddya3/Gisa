import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


warnings.filterwarnings(action = 'ignore')
pd.set_option('max_column', 20)

# 데이터 수집
x_train = pd.read_csv("gisa/bigData-main/service_X_train.csv")
x_test = pd.read_csv("gisa/bigData-main/service_X_test.csv")
y_train = pd.read_csv("gisa/bigData-main/service_y_train.csv")


# 라벨인코딩, 스케일링, 공선성

# 다중공선성 제거
# print(x_train.corr())

# 필요없는 열 제거
x_test_id = x_test['CustomerId']
x_train.drop(columns=['CustomerId', 'Surname'], inplace=True)
x_test.drop(columns=['CustomerId', 'Surname'], inplace=True)
y_train.drop(columns=['CustomerId'], inplace=True)

# 라벨인코딩 - gender
x_train['Gender'].replace('Female', 'female', inplace=True)
x_train['Gender'].replace('Male', 'male', inplace=True)
x_train['Gender'] = x_train['Gender'].map(lambda x : 0 if x == 'female' else 1)

x_test['Gender'].replace('Female', 'female', inplace=True)
x_test['Gender'].replace('Male', 'male', inplace=True)
x_test['Gender'] = x_test['Gender'].map(lambda x : 0 if x == 'female' else 1)

# 라벨인코딩 - geography
encoder = LabelEncoder()
x_train['Geography'] = encoder.fit_transform(x_train['Geography'])
x_test['Geography'] = encoder.transform(x_test['Geography'])

# 스케일링
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x_test.columns)

# train, val 나누기
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)

# 모델링
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, Y_train)

Y_predict = model.predict(X_val)
Y_predict = pd.DataFrame(Y_predict, columns = ['Exited'])

print(roc_auc_score(Y_predict, Y_val))      #84.41%

# 제출
y_predict = model.predict_proba(x_test)
y_predict = pd.DataFrame(y_predict[:, 1], columns = ['Exited'])

finaldf = pd.concat([x_test_id, y_predict], axis=1)
finaldf.to_csv('gisa/bigData-main/service_final.csv', index=False)


# 채점
y_test = pd.read_csv("gisa/bigData-main/service_y_test.csv")
print(roc_auc_score(y_test['Exited'], y_predict))