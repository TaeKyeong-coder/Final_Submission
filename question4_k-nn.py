#knn사용을 위한 라이브러리
from sklearn.neighbors import KNeighborsClassifier
#모델 생성 및 평가를 위한 sklearn LogisticRegression 사용하기 위해 import
from sklearn.linear_model import LogisticRegression
#학습세트/평가세트 분리를 위한 train_test_split 사용
from sklearn.model_selection import train_test_split
#데이터 정규화를 위한 StandardScaler를 사용
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

#사용할 기반 data 파일 불러오기(제 python버전에서는 encoding값을 안 주면 코드가 안 돌아가서 넣었지만 지워도 됩니다.)
subway = pd.read_csv("subway_2.csv", encoding="cp949")


#승차/하차총승객수로 데이터 세트 준비.
features = subway[['승차총승객수', '하차총승객수']]
sub_n = subway[['노선명']]

#학습세트/평가세트 분리하기
training_data, validation_data , training_labels, validation_labels = train_test_split(features, sub_n, test_size = 0.2, random_state = 100)

#데이터 정규화(스케일링)하기
scaler = StandardScaler()
train_features = scaler.fit_transform(training_data)
test_features = scaler.transform(validation_data)

classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(training_data, training_labels)

#예측
print("k-nn으로 예측")

#데이터 넣어서 예측해보기
my_data = np.array([30000, 30000])
subway_predict = np.array([my_data])
subway_predict = scaler.transform(subway_predict)
print(classifier.predict(subway_predict))

#정확도
print("정확도는")
print(classifier.score(validation_data, validation_labels))
