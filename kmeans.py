import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans
from sklearn import preprocessing 
import pandas as pd 

df = pd.read_excel('~/Downloads/titanic.xls')
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)


def convert_non_numeric(df):
	columns = df.columns.values

	for column in columns:
		text_digit_values = {}

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			contents = df[column].values.tolist()
			unique_elements = set(contents)
			x=0
			for i in unique_elements:
				if i not in text_digit_values:
					text_digit_values[i] = x
					x+=1

			df[column] = list(map(lambda x:text_digit_values[x],df[column]))
	return df 


df = convert_non_numeric(df)

df.drop(['sex','boat'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0

for i in range(len(X)):
	prediction = np.array(X[i].astype(float))
	prediction = prediction.reshape(-1,len(prediction))

	if clf.predict(prediction) == y[i]:
		correct+=1


print(correct/len(X))




