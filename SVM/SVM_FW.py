from sklearn.model_selection import train_test_split 
from sklearn import svm
import pandas as pd 



df = pd.read_csv('../datasets/breast_cancer.csv')

df.drop(['id','Unnamed: 32'],inplace=True,axis=1)

total_data = df.values.tolist()

data = []
labels = []

for i in total_data:
    data.append(i[1:])
    labels.append(i[0])


X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.4, random_state=0)


clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train,y_train)

score = clf.score(X_test,y_test)

print(score)
