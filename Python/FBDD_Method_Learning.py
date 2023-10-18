import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd                                              #ver. 1.5.3
import numpy as np                                               #ver. 1.23.5
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso

def LassoRanking(X, y):
    lasso = Lasso(alpha=0.001)

    scores = []
    scores.append(np.argsort(lasso.fit(X, y).coef_))

    return scores[0]

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(loadarff(fileName)[0])
    i = len(dataFrame.columns)-1
    col = classLabelEncoder.fit_transform(dataFrame.iloc[:, i].values)
    dataFrame.isetitem(i, col)
    return dataFrame

def computeSTDinChunk(chunk, records_in_chunk, debug):
    numberOfChunks = int(chunk.shape[0] / records_in_chunk)
    split = np.array_split(chunk, numberOfChunks)
    results = []
    featuresNumber = len(chunk.columns) - 1
    places = [0] * featuresNumber
    if debug == 1:
        f.write("******************************\n")
    for i in range(0, len(split)):
        X = split[i].drop(columns="class")
        y = split[i]["class"]

        results.append(LassoRanking(np.array(X), np.array(y)))

        if debug == 1:
            f.write(np.array2string(results[i]))
            f.write("\n")

        for j in range(0, len(places)):
            pos = np.where(results[i] == j)
            places[j] += pos[0][0]

    #feat = np.argmin(places)

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
    if debug == 1:
        f.write("******************************\n")
        f.write("Feature   = %d\n" % feat[0])
        f.write("AVR       = %2.2f\n" % feat[1])
        f.write("STD       = %2.2f\n" % feat[2])
        f.write("Threshold = %2.2f\n" % feat[3])
        f.write("******************************\n")

    return (feat[0], feat[3])

debug = 0

if debug == 1:
    f = open("_Results.txt", "w")

inputPath = "Benchmarks/"
fileName = "spam.arff"

recordsInChunk =500
dataFrame = loadArffData(inputPath + fileName)
numberOfChunk = int(np.ceil(dataFrame.shape[0] / recordsInChunk))
dataSplit = np.array_split(dataFrame, numberOfChunk)

(feat0, threshold) = computeSTDinChunk(dataSplit[0], recordsInChunk / 10, debug)

# Train RandomForest classifier by chunk number 0.
X_train = dataSplit[0].drop(columns="class")
y_train = dataSplit[0]["class"]
#y_train = y_train.astype('int')
classifier = RandomForestClassifier(random_state=0)

classifier.fit(X_train, y_train)

elapsed_time = 0
st = time.time()
drifts = []
results = []
feat = []
for i in range(1, numberOfChunk):
    X = dataSplit[i].drop(columns = "class")
    y = dataSplit[i]["class"]
    y = y.astype('int')
    y_predict = classifier.predict(X)
    result = accuracy_score(y, y_predict)
    results.append(result)

    featuresNumber = len(X.columns)
    instancesNumber = len(X)
    score = LassoRanking(np.array(X), np.array(y))
    pos = np.where(score == feat0)
    if pos[0][0] > threshold:
        drifts.append(i)
        # Re-learning
        chunk = i+1
        if chunk < numberOfChunk:
            X_train = dataSplit[chunk].drop(columns = "class")
            y_train = dataSplit[chunk]["class"]
            y_train = y_train.astype('int')
            classifier.fit(X_train, y_train)
            (feat0, threshold) = computeSTDinChunk(dataSplit[chunk], recordsInChunk / 10, debug)
et = time.time()
elapsed_time = et - st

if debug == 1:
    f.close()

print("Accuracy :",round(np.mean(results),3))
print("Number of drifts : ", len(drifts))
print("Time :", round(elapsed_time,3))
print("Done!")

if len(drifts) > 0:
    plt.axvline(x = (drifts[0] - 1), color = 'b', linewidth=2)
    for i in range(1, len(drifts)):
        plt.axvline(x = (drifts[i] - 1), color = 'b', linewidth=2)
# plt.plot(range(0, len(results)), feat, 'g', linewidth=1)
plt.plot(range(0, len(results)), results, 'r', linewidth=1)
plt.xlabel('Chunks', fontsize=12)
#plt.ylabel('Accuracy (red) / Feature rank (green)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
#plt.legend()
plt.axis([0,len(results),0,1.05])
plt.grid(True)
#plt.savefig("Figure_" + str(recordsInChunk) + ".pdf")
plt.show()
