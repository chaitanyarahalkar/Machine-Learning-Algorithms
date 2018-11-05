import numpy as np 
import pandas as pd 
import random 
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn import svm

def k_nearest_neighbors(train_set,data,k=3):
    if len(train_set)>= k:
        print('set k to be less than the total data')
    distances = []

    for group in train_set:
        for features in train_set[group]:
            eu_distance = np.linalg.norm(np.array(features)-np.array(data))
            distances.append([eu_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result
df = pd.read_csv('~/Downloads/data.csv')

df.drop(['id','Unnamed: 32'],inplace=True,axis=1)

total_data = df.values.tolist()

data = []
labels = []

for i in total_data:
    data.append(i[1:])
    labels.append(i[0])

X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.4, random_state=0)
random.shuffle(total_data)

test_size = 0.4 

train_data = total_data[:-int(test_size*len(total_data))]
test_data = total_data[-int(test_size*len(total_data)):]


train_set = {'M':[],'B':[]}
test_set = {'M':[],'B':[]}

for i in train_data:
    train_set[i[0]].append(i[1:])

for i in test_data:
    test_set[i[0]].append(i[1:])


correct = 0 
total = 0 

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set,data,k=5)

        if vote == group:
            correct+=1

        total+=1


print('Accuracy:',correct/total)


clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train,y_train)

score = clf.score(X_test,y_test)

print(score)
