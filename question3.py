import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("subway_1.csv", encoding = "cp949")

X = np.array(df[['승차총승객수', '하차총승객수']])
y = np.array(df['노선명'])

clf = LinearDiscriminantAnalysis()
clf.fit(X,y)

clf.predict([[10000,10000]])
