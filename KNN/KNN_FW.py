import pandas as pd 
from sklearn import preprocessing,cross_validation,neighbors



df = pd.read_csv('../datasets/breast_cancer.csv')

df.drop(['id','Unnamed: 32'],inplace=True,axis=1)

total_data = df.values.tolist()

data = []
labels = []

for i in total_data:
    data.append(i[1:])
    labels.append(i[0])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(data,labels, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)