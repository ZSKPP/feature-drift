import numpy as np
import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(arff.loadarff(fileName)[0])
    for i in range(0, len(dataFrame.columns)):
        col = classLabelEncoder.fit_transform(dataFrame.iloc[:,i].values)
        dataFrame.iloc[:, i] = col
    return dataFrame

RECORDS_IN_CHUNK = 1000  # 1000 lub 2300

# Load data *.arff and split data into chunk.
dataFrame = loadArffData("abrupt_HP_10.arff")
numberOfChunk = int(dataFrame.shape[0] / RECORDS_IN_CHUNK)
dataSplit = np.array_split(dataFrame, numberOfChunk)

#numberOfColumns = len(dataFrame.columns) - 1
# Train Classifier by chunk number 0.
X_train = dataSplit[0].drop(columns="class")
print()
y_train = dataSplit[0]["class"]
print(X_train,y_train)
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

#Classifier
typeOfClass = RandomForestClassifier(random_state=0)
#typeOfClass = AdaBoostClassifier()
#typeOfClass = GaussianNB()
#typeOfClass = tree.DecisionTreeClassifier()

typeOfClass.fit(X_train, y_train)

results = []
for i in range(1, numberOfChunk):
    X = dataSplit[i].drop(columns = "class")
    y = dataSplit[i]["class"]
    y_predict = typeOfClass.predict(X)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    accuracy = accuracy_score(y, y_predict)
    results.append(accuracy)
    print('chunk=',i,'Acc=',accuracy)

#alternative calculation:
#avg_ACC=sum(results)/len(results)

avg_ACC=np.mean(results)
avg_ACC=round(avg_ACC,3)
print('avg_ACC=',avg_ACC)
plt.plot(range(0, len(results)), results, 'r', linewidth=1)
plt.xlabel('Chunks', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axis([0,len(results),0,1.05])
plt.grid(True)
plt.show()

print("Done!")

