import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd 
from sklearn import preprocessing

class KMeans:
	def __init__(self,k=2,tol=0.001,max_iterations=300):
		self.k = k 
		self.tol = tol 
		self.max_iterations = max_iterations 

	def fit(self,data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iterations):
			self.classifications = dict()

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)


			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)


			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]

				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
					optimized = False


				if optimized:
					break 



	def predict(self,data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

df = pd.read_excel('../datasets/titanic.xls')
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

df.drop(['home.dest','ticket','sex'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])


clf = KMeans()
clf.fit(X)


correct = 0
for i in range(len(X)):

    prediction = np.array(X[i].astype(float))
    prediction = prediction.reshape(-1, len(prediction))
    prediction = clf.predict(prediction)
    if prediction == y[i]:
        correct += 1


print(correct/len(X))
