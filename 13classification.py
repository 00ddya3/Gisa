from operator import contains, mod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings(action='ignore')


# 데이터 불러오기
x_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/titanic_x_train.csv', encoding='cp949')
x_test = pd.read_csv('D:/python_workspace/gisa/bigData-main/titanic_x_test.csv', encoding='cp949')
y_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/titanic_y_train.csv')


# 불필요한 컬럼 제거
x_test_id = x_test['PassengerId']
x_train.drop(columns=['PassengerId'], axis=1, inplace=True)
x_test.drop(columns=['PassengerId'], axis=1, inplace=True)
y_train.drop(columns=['PassengerId'], axis=1, inplace=True)


# unique한 값의 개수 확인
#print(x_test.nunique())


# 두 데이터 합쳐서 같이 전처리
x_train['label'] = 'train'
x_test['label'] = 'test'
x_data = pd.concat([x_train, x_test], axis=0)


# 결측치 처리 - 나이
bands = {'Mr', 'Mrs', 'Miss', 'Ms'}

x_data['band'] = 'other'
for band in bands :
    x_data.loc[x_train['승객이름'].str.contains(band),'band'] = band        # 승객이름에 band가 포함되면 band열에 그 값 표시

ageseries = x_data.groupby('band').mean()
ageseries = round(ageseries['나이'], 1)
agedict = ageseries.to_dict()       # key: band / value: 평균나이

x_data['band'] = x_data['band'].map(agedict)        # band열을 평균나이로 대체
x_data['나이'] = x_data['나이'].fillna(x_data['band'])      # 나이열이 결측치면 평균나이로 대체


# 결측치 처리 - 객실번호
x_data.drop(columns = ['객실번호'], inplace=True)       #생존과 상관이 적으므로 제거


# 결측치 처리 - 선착장
#print(x_data['선착장'].describe())
x_data['선착장'] = x_data['선착장'].fillna('S')     # S선착장이 제일 많았음


# 불필요한 컬럼 제거
x_data.drop(columns=['승객이름', '티켓번호', 'band'], inplace=True)


# 범주형 변수 인코딩 - 성별
x_data['성별'] = x_data['성별'].apply(lambda x : 0 if x == 'male' else 1)


# 범주형 변수 인코딩 - 선착장
#dummy = pd.get_dummies(x_data['선착장'], drop_first=True)
encoder = LabelEncoder()
x_data['선착장'] = encoder.fit_transform(x_data['선착장'])


# 파생변수 만들기
x_data['가족수'] = x_data['형제자매배우자수'] + x_data['부모자식수']
x_data.drop(columns=['형제자매배우자수', '부모자식수'], inplace=True)


# train,test 다시 나눠주기
x_train = x_data[x_data['label'] == 'train']
x_test = x_data[x_data['label'] == 'test']

x_train.drop(columns=['label'], inplace=True)
x_test.drop(columns=['label'], inplace=True)


# train데이터를 train과 validation 으로 나눠주기
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)


# 모델링
#model = XGBClassifier(eval_metric = 'error', random_state=10)       #기본모델 (roc: 0.7867)
model = XGBClassifier(eval_metric = 'error', random_state=10, n_estimators=100, max_depth=5)    #하이퍼파라미터 튜닝 (roc: 0.8559)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_val)
Y_predict = pd.DataFrame(Y_predict, columns=['Survived'])


# 모델 평가하기
print(roc_auc_score(Y_val, Y_predict))


# 실제 테스트 데이터로 진행
y_predict = model.predict(x_test)
y_predict = pd.DataFrame(y_predict, columns=['Survived'])


# 결과 제출하기
finaldf = pd.concat([x_test_id, y_predict], axis=1)
finaldf.to_csv('D:/python_workspace/gisa/bigData-main/titanic_final.csv', index=False)      #index=False 필수!!