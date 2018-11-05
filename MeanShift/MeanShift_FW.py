import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_excel('../datasets/titanic.xls')

original_df = pd.DataFrame.copy(df)

df.drop(['body','name'],1,inplace=True)

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

df.drop(['ticket','home.dest'],1)

X = np.array(df.drop(['survived'],1).astype(float)) 
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)


labels = clf.labels_
cluster_centers = clf.cluster_centers_ 

original_df['cluster_group']=np.nan 

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]


n_clusters_ = len(np.unique(labels))
survival_rates = dict()


for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]

    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    survival_rate = len(survival_cluster) / len(temp_df)
    
    survival_rates[i] = survival_rate
    
print(survival_rates)