from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#scatter plot그리기 위해서 import
import matplotlib.pyplot as plt

df = pd.read_csv("nofare1.csv", encoding = "cp949")

#단순 데이터 확인용입니다. 문제와는 상관 없습니다.
df.head()

x = df[" 총승차 "]
y = df[" 총하차 "]

#LinearRegression적
line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1,1), y)

#예측하기
print("총하차인원은")
print(line_fitter.predict([[10000]]))

#line그래프 먼저 뜹니다. 그 다음 (끄면) scatter 확인가능 합니다.
#Basic Line Plot
plt.plot(x,y)
plt.show()

#scatter plot
#잘 안 보여서 투명도 조절하고 점 크기 등 몇 개 조절했습니다.
#그냥 기본 scatter plot보시려면 plot(x,y)만 남겨두고 뒤에는 지워주시면 됩니다.
plt.plot(x,y,'or', 'MarkerSize',5, alpha=0.5)
plt.show()

