#모델 생성 및 평가를 위한 sklearn LogisticRegression 사용하기 위해 import
from sklearn.linear_model import LogisticRegression
#학습세트/평가세트 분리를 위한 train_test_split 사용
from sklearn.model_selection import train_test_split
#데이터 정규화를 위한 StandardScaler를 사용
from sklearn.preprocessing import StandardScaler
#qda 사용하기 위한 라이브러리
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pandas as pd
import numpy as np

#사용할 기반 data 파일 불러오기(제 python버전에서는 encoding값을 안 주면 코드가 안 돌아가서 넣었지만 지워도 됩니다.)
subway = pd.read_csv("subway_2.csv", encoding="cp949")


#몇 호선인지 도출하기 위한 값으로 승차/하차총승객수를 feature로 고르고 데이터 세트 준비.
features = subway[['승차총승객수', '하차총승객수']]
sub_n = subway[['노선명']]

#학습세트/평가세트 분리하기
train_features, test_features, train_labels, test_labels = train_test_split(features, sub_n)

#데이터 정규화(스케일링)하기
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#LogisticRegression 모델 생성
model = LogisticRegression()

#features, labels을 fit시킨다.
model.fit(train_features, train_labels)

print("logisticRegression으로 예측")

#데이터 넣어서 예측해보기
my_data = np.array([30000, 30000])
subway_predict = np.array([my_data])
subway_predict = scaler.transform(subway_predict)
print(model.predict(subway_predict))

#정확도
print("정확도는")
print(model.score(train_features, train_labels))

#qda구현
X = np.array(subway[['승차총승객수', '하차총승객수']])
y = np.array(subway['노선명'])

clf = QuadraticDiscriminantAnalysis()
clf.fit(X,y)

print("QDA로 예측")
print(clf.predict([[30000, 30000]]))
print("정확도는")
print(clf.score(train_features, train_labels))
