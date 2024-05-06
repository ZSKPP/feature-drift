# Python interpreter 3.8
# -----------------------
# Pandas       -> ver. 1.5.3
# Numpy        -> ver. 1.23.5
# Scipy        -> ver. 1.10.1
# Scikit-learn -> ver. 1.2.2
# MatplotLib   -> ver. 3.7.5

import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

def LassoRanking(X, y):
    lasso = Lasso(alpha=0.001)

    scores = []
    scores.append(np.argsort(lasso.fit(X, y).coef_))

    return scores[0]

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(loadarff(fileName)[0])
    for column in dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(dataFrame[column]):
            dataFrame[column] = classLabelEncoder.fit_transform(dataFrame[column])
    return dataFrame

def computeSTDinChunk(chunk, records_in_chunk):
    numberOfChunks = int(chunk.shape[0] / records_in_chunk)
    split = np.array_split(chunk, numberOfChunks)
    results = []
    featuresNumber = len(chunk.columns) - 1
    places = [0] * featuresNumber
    for i in range(0, len(split)):
        X = split[i].drop(columns="class")
        y = split[i]["class"]

        results.append(LassoRanking(np.array(X), np.array(y)))

        for j in range(0, len(places)):
            pos = np.where(results[i] == j)
            places[j] += pos[0][0]

    featureRanking = []
    for j in range(0, len(places)):
        featPlace = []
        for i in range(0, len(results)):
            featPlace.append(np.where(results[i] == j)[0][0])
        threshold = np.mean(featPlace) + np.std(featPlace)
        threshold = np.ceil(threshold)
        featureRanking.append((j, np.mean(featPlace), np.std(featPlace), threshold))

    #feat = np.argmin(places)
    feat = min(featureRanking, key = lambda t: t[1])
    return (feat[0], feat[3])

def detectDriftWithoutRetrainingClassifier(classifier, dataSplit, numberOfChunk, feat0, threshold):
    drifts = []
    results = []
    for i in range(1, numberOfChunk):
        X = dataSplit[i].drop(columns = "class")
        y = dataSplit[i]["class"]
        y_predict = classifier.predict(X)
        result = accuracy_score(y, y_predict)
        results.append(result)
        
        score = LassoRanking(np.array(X), np.array(y))
        pos = np.where(score == feat0)
        
        if pos[0][0] > threshold:
            drifts.append(i)
            if (i + 1) < numberOfChunk:
                (feat0, threshold) = computeSTDinChunk(dataSplit[i + 1], recordsInChunk / 10)

    return (results, drifts)

def detectDriftWithRetrainingClassifier(classifier, dataSplit, numberOfChunk, feat0, threshold):
    drifts = []
    results = []
    for i in range(1, numberOfChunk):
        X = dataSplit[i].drop(columns = "class")
        y = dataSplit[i]["class"]
        y = y.astype('int')
        y_predict = classifier.predict(X)
        result = accuracy_score(y, y_predict)
        results.append(result)
        
        score = LassoRanking(np.array(X), np.array(y))
        pos = np.where(score == feat0)
        if pos[0][0] > threshold:
            drifts.append(i)
            # Re-learning
            chunk = i + 1
            if chunk < numberOfChunk:
                X_train = dataSplit[chunk].drop(columns = "class")
                y_train = dataSplit[chunk]["class"]
                y_train = y_train.astype('int')
                classifier.fit(X_train, y_train)
                (feat0, threshold) = computeSTDinChunk(dataSplit[chunk], recordsInChunk / 10)
    
    return (results, drifts)

inputPath = "Benchmarks/"
fileName = "spam.arff"

recordsInChunk = 500

dataFrame = loadArffData(inputPath + fileName)
numberOfChunk = int(np.ceil(dataFrame.shape[0] / recordsInChunk))
dataSplit = np.array_split(dataFrame, numberOfChunk)

(feat0, threshold) = computeSTDinChunk(dataSplit[0], recordsInChunk / 10)

# Train RandomForest classifier by chunk number 0.
X_train = dataSplit[0].drop(columns="class")
y_train = dataSplit[0]["class"]

classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

st = time.time()
(accuracyWithoutRetrainingClassifier, driftsWithoutRetrainingClassifier) = \
    detectDriftWithoutRetrainingClassifier(classifier, dataSplit, numberOfChunk, feat0, threshold)
et = time.time()
timeWithoutRetrainingClassifier = et - st

st = time.time()
(accuracyWithRetrainingClassifier, driftsWithRetrainingClassifier) = \
    detectDriftWithRetrainingClassifier(classifier, dataSplit, numberOfChunk, feat0, threshold)
et = time.time()
timeWithRetrainingClassifier = et - st

print("-")
print("Average accuracy without retraining classifier: ", round(np.mean(accuracyWithoutRetrainingClassifier), 3))
print("Number of drifts without retraining classifier: ", len(driftsWithoutRetrainingClassifier))
print("Time without retraining classifier: ", round(timeWithoutRetrainingClassifier,3))
print("-")
print("Average accuracy with retraining classifier: ", round(np.mean(accuracyWithRetrainingClassifier), 3))
print("Number of drifts with retraining classifier: ", len(driftsWithRetrainingClassifier))
print("Time with retraining classifier: ", round(timeWithRetrainingClassifier,3))
print("-")

plt.subplot(2,1,1)
if len(driftsWithoutRetrainingClassifier) > 0:
    plt.axvline(x = (driftsWithoutRetrainingClassifier[0] - 1), color = 'b', linewidth=2)
    for i in range(1, len(driftsWithoutRetrainingClassifier)):
        plt.axvline(x = (driftsWithoutRetrainingClassifier[i] - 1), color = 'b', linewidth=2)
plt.plot(range(0, len(accuracyWithoutRetrainingClassifier)), accuracyWithoutRetrainingClassifier, 'r', linewidth=1)
plt.xlabel('Chunks', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axis([0,len(accuracyWithoutRetrainingClassifier),0,1.1])
plt.grid(True)

plt.subplot(2,1,2)
if len(driftsWithRetrainingClassifier) > 0:
    plt.axvline(x = (driftsWithRetrainingClassifier[0] - 1), color = 'b', linewidth=2)
    for i in range(1, len(driftsWithRetrainingClassifier)):
        plt.axvline(x = (driftsWithRetrainingClassifier[i] - 1), color = 'b', linewidth=2)
plt.plot(range(0, len(accuracyWithRetrainingClassifier)), accuracyWithRetrainingClassifier, 'r', linewidth=1)
plt.xlabel('Chunks', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axis([0,len(accuracyWithRetrainingClassifier),0,1.1])
plt.grid(True)

plt.show()
