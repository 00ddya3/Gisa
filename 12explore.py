import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings(action='ignore')

x_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/x_train.csv', encoding='cp949')
x_test = pd.read_csv('D:/python_workspace/gisa/bigData-main/x_test.csv', encoding='cp949')
y_train = pd.read_csv('D:/python_workspace/gisa/bigData-main/y_train.csv', encoding='cp949')

#불필요한 컬럼 삭제
x_test_custid = x_test['cust_id']       # 하나 따로 저장해두기
x_train.drop(columns=['cust_id'], inplace=True)
x_test.drop(columns=['cust_id'], inplace=True)
y_train.drop(columns=['cust_id'], inplace=True)

#결측치 처리
x_train['환불금액'].fillna(0, inplace=True)
x_test['환불금액'].fillna(0, inplace=True)

#범주형 변수 인코딩
encoder = LabelEncoder()
x_train['주구매상품'] = encoder.fit_transform(x_train['주구매상품'])
x_train['주구매지점'] = encoder.fit_transform(x_train['주구매지점'])
x_test['주구매상품'] = encoder.fit_transform(x_test['주구매상품'])
x_test['주구매지점'] = encoder.fit_transform(x_test['주구매지점'])
#print(encoder.classes_)        #라벨 인코딩 변환결과

#파생변수만들기
con = x_train['환불금액'] > 0

x_train['환불여부'] = 0
x_train.loc[con, '환불여부'] = 1
x_train.drop(columns=['환불금액'], inplace=True)

x_test['환불여부'] = 0
x_test.loc[con, '환불여부'] = 1
x_test.drop(columns=['환불금액'], inplace=True)

#표준화 스케일링
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns= x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns= x_test.columns)

#상관관계 확인하기
#print(x_train.corr())       #총구매액과 최대구매액의 상관관계가 0.7 높게 나옴
x_train.drop(columns = ['최대구매액'], inplace=True)    #다중공선성을 제거하기 위해 삭제
x_test.drop(columns = ['최대구매액'], inplace=True)     #다중공선성을 제거하기 위해 삭제

#의사결정나무 모델 + 하이퍼파라미터 튜닝
model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=10)
model.fit(x_train, y_train)                 #모델학습
y_predict = model.predict(x_test)           #학습된 모델으로 테스트 데이터 예측하기
y_predict = pd.DataFrame(y_predict)

#모델 평가하기
#print(roc_auc_score(y_predict, y_test))    #y_test가 없어서 실행안됨

#결과 제출하기
finaldf = pd.concat([x_test_custid, y_predict], axis=1)
finaldf = finaldf.rename(columns = {0 : 'gender'})
finaldf.to_csv('D:/python_workspace/gisa/bigData-main/final.csv', index=False)      #index=False 필수!!