#문제 내 알려준 라이브러리
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#파일 열
df = pd.read_csv("student_health_5.csv", encoding = "cp949")

#키와 체중을 입력값으로 하는 클러스터링 훈련시키고
processed_data = df.copy()
scaler = preprocessing.MinMaxScaler()
processed_data[['키', '몸무게']]=\
                     scaler.fit_transform(processed_data[['키', '몸무게']])


#K=2로 클러스터링
kmeans = KMeans(n_clusters=2).fit_predict(processed_data[['키', '몸무게']])

#저학년 고학년 분류
results = np.array(processed_data['학년'])
for idx in range(0,len(results)):
       if results[idx] >= 4:
              #고학년 전부 0이라는 값의 클래스로 처리
              results[idx] = 0
              cnt = 0
              
#훈련된 모델의 정확률을 계산하는 파이썬 코드를 작성하시오.
y = results

for idx in range(0,len(results)):
       if results[idx] == y[idx]:
              cnt += 1

print("Accuracy: ", cnt/len(results))
