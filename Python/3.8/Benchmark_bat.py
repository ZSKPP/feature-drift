# Python interpreter 3.8
# ----------------------
# Pandas       -> ver. 1.5.3
# Numpy        -> ver. 1.23.5
# Scikit-learn -> ver. 1.3.0
# Frouros      -> ver. 0.6.1
# MatplotLib   -> ver. 3.7.5

import warnings
warnings.filterwarnings('ignore')
import os
import csv
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from frouros.metrics import PrequentialError
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

# Concept drift / Streaming / Change detection
from frouros.detectors.concept_drift import BOCD, BOCDConfig
from frouros.detectors.concept_drift import CUSUM, CUSUMConfig
from frouros.detectors.concept_drift import GeometricMovingAverage, GeometricMovingAverageConfig
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig

# Concept drift / Streaming / Statistical process control
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.concept_drift import ECDDWT, ECDDWTConfig
from frouros.detectors.concept_drift import EDDM, EDDMConfig
from frouros.detectors.concept_drift import HDDMA, HDDMAConfig
from frouros.detectors.concept_drift import HDDMW, HDDMWConfig
from frouros.detectors.concept_drift import RDDM, RDDMConfig

# Concept drift / Streaming / Window based
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.concept_drift import KSWIN, KSWINConfig
from frouros.detectors.concept_drift import STEPD, STEPDConfig

def loadData(fileName):
    # Load data from .arff or .csv file to DataFrame.
    if os.path.splitext(fileName)[1] == ".arff":
        dataFrame = pd.DataFrame(loadarff(fileName)[0])
    if os.path.splitext(fileName)[1] == ".csv":
        dataFrame = pd.read_table(fileName, delimiter=str(csv.Sniffer().sniff(open(fileName, 'r').read()).delimiter))
    # Encoding to numeric type.
    classLabelEncoder = LabelEncoder()
    for column in dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(dataFrame[column]):
            dataFrame[column] = classLabelEncoder.fit_transform(dataFrame[column])
    return dataFrame

def trainAndTest(classifier, dataFrame, position, trainingSamples):
    training = dataFrame.iloc[position: position + trainingSamples, :]
    X_train = training.drop(columns = "class").to_numpy()
    y_train = training["class"].astype('int')
    
    classifier.fit(X_train, y_train)
    
    tests = dataFrame.iloc[position + trainingSamples:, :]
    X = tests.drop(columns = "class")
    y = tests["class"].to_numpy().astype('int')
    
    y_predict = classifier.predict(X)
    
    return (y, y_predict)

def AccuracyWithoutDrifts(classifier, dataFrame, trainingSamples):
    metric = PrequentialError(alpha = 1.0)
    accuracy = []
    (y, y_predict) = trainAndTest(classifier, dataFrame, 0, trainingSamples)
    # Assume: len(y) = len(y_predict)
    for i in range(0, len(y)):
        if (y[i:(i + 1)] == y_predict[i:(i + 1)]):
            error = 1
        else:
            error = 0
        metric_error = metric(error_value=error)
        accuracy.append(metric_error)
    
    return accuracy

def findDrift(y, y_predict, y_all, y_predict_all, driftDetector):
    # Assume: len(y) = len(y_predict)
    metric = PrequentialError(alpha = 1.0)
    i = 0
    while i < len(y):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        if (y[i:(i + 1)] == y_predict[i:(i + 1)]):
            error = 1
        else:
            error = 0
        metric_error = metric(error_value=error)
        _ = driftDetector.update(value=error)
        i = i + 1
        status = driftDetector.status
        if status["drift"]:
            driftDetector.reset()
            return i

    return -1

def completeAccuracyArray(y, y_predict, y_all, y_predict_all, i, trainingSamples):
    k = 0
    while (k < trainingSamples) and (i < len(y)):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        i = i + 1
        k = k + 1

def AccuracyWithDrifts(classifier, driftDetector, dataFrame, trainingSamples):
    j = 0
    t = 0
    drft = 0
    finish = False
    drifts = []
    y_all = []
    y_predict_all = []
    while not finish:
        if j + trainingSamples < len(dataFrame):
            (y, y_predict) = trainAndTest(classifier, dataFrame, j, trainingSamples)
            i = findDrift(y, y_predict, y_all, y_predict_all, driftDetector)
            if (i == -1):
                finish = True
            else:
                drft = drft + i + t
                drifts.append(drft)
                j = drft + trainingSamples
                completeAccuracyArray(y, y_predict, y_all, y_predict_all, i, trainingSamples)
            t = trainingSamples
        else:
            finish = True
    
    accuracy = []
    # Assume: len(y_all) = len(y_predict_all)

    metric = PrequentialError(alpha = 1.0)
    for i in range(0, len(y_all)):
        if (y_all[i:(i + 1)] == y_predict_all[i:(i + 1)]):
            error = 1
        else:
            error = 0
        metric_error = metric(error_value = error)
        accuracy.append(metric_error)
    
    return (accuracy, drifts)

def DrawChart(accuracyWithoutDrifts, accuracyWithDrifts, drifts):
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithoutDrifts)), accuracyWithoutDrifts, 'r')
    plt.xlabel('Sample', fontsize = 12)
    plt.ylabel('Accuracy')
    plt.tick_params(axis = 'both', which = 'major')
    plt.grid(True)
    # plt.savefig("WithoutDrift.pdf")
    plt.show()
    
    plt.clf()
    if len(drifts) > 0:
        plt.axvline(x = (drifts[0] - 1), color = 'b', linewidth = 2)
        for i in range(1, len(drifts)):
            plt.axvline(x = (drifts[i] - 1), color = 'b', linewidth = 2)
            plt.axvline(x = (drifts[i] + trainingSamples - 1), color = 'g', linewidth = 2)
    
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithDrifts)), accuracyWithDrifts, 'r')
    plt.xlabel('Sample', fontsize = 12)
    plt.ylabel('Accuracy')
    plt.tick_params(axis = 'both', which = 'major')
    plt.grid(True)
    # plt.savefig("WithDrift.pdf")
    plt.show()

##################################################################################
# Main()

clf = RandomForestClassifier(random_state = 0)

########################################################

##### Concept drift / Streaming / Change detection
bocd = BOCD(config = BOCDConfig())
cusum = CUSUM(config = CUSUMConfig())
gma = GeometricMovingAverage(config = GeometricMovingAverageConfig())
ph = PageHinkley(config = PageHinkleyConfig())

##### Concept drift / Streaming / Statistical process control
ddm = DDM(config = DDMConfig())
ecddwt = ECDDWT(config = ECDDWTConfig())
eddm = EDDM(config = EDDMConfig())
hddma = HDDMA(config = HDDMAConfig())
hddmw = HDDMW(config = HDDMWConfig())
rddm = RDDM(config = RDDMConfig())

##### Concept drift / Streaming / Window based
adwin = ADWIN(config = ADWINConfig())
kswin = KSWIN(config = KSWINConfig())
stepd = STEPD(config = STEPDConfig())

########################################################

detectors = [ddm, eddm, kswin, ph, stepd, adwin]
detectorsName = ["ddm", "eddm", "kswin", "ph", "stepd", "adwin"]

fileNames = ["Digits08.arff", "Digits17.arff", "Wine.arff", "Spam.arff", "Phishing.arff",
             "CovType.arff", "Electricity.arff", "Pokerhand.arff", "Insect.arff"]
trainingSamples = [150, 150, 500, 500, 500, 2500, 2500, 55000, 2900]

#fileNames = ["Abrupt_HP_5.arff", "Abrupt_HP_10.arff", "Gradual_HP_5.arff", "Gradual_HP_10.arff",
#             "Recurring_HP_5.arff", "Recurring_HP_10.arff", "Incremental_HP_5.arff", "Incremental_HP_10.arff"]
#trainingSamples = [2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300]

inputPath = "Benchmarks/"

f = open("Results.txt", "w")

f.write('Dataset & ')
for i in range(0, len(detectorsName)):
    if i == len(detectorsName) - 1:
        f.write('%s' % (detectorsName[i]))
    else:
        f.write('%s & ' % (detectorsName[i]))
f.write(" \u005c\u005c")
f.write(" \hline")

f.write("\n")
f.write("\n")

for j in range(0, len(fileNames)):
    print(fileNames[j], end =" ")

    dataFrame = loadData(inputPath + fileNames[j])
    fileName = Path(fileNames[j]).stem
    f.write('%s & ' % (fileName))

    accuTable = []
    driftsTable = []
    for i in range(0, len(detectors)):
        print("*", end =" ")
        (accuracyWithDrifts, drifts) = AccuracyWithDrifts(clf, detectors[i], dataFrame, trainingSamples[j])
        accuTable.append(accuracyWithDrifts[-1])
        driftsTable.append(len(drifts))
        detectors[i].reset()

    for i in range(0, len(detectors)):
        if i == len(detectors) - 1:
            f.write('%1.3f \u005c\u005c' % accuTable[i])
        else:
            f.write('%1.3f & ' % accuTable[i])
    f.write("\n")
    f.write(" & ")
    for i in range(0, len(detectors)):
        if i == len(detectors) - 1:
            f.write('%d/%d \u005c\u005c' % (trainingSamples[j], driftsTable[i]))
        else:
            f.write('%d/%d & ' % (trainingSamples[j], driftsTable[i]))
    f.write(" \hline")

    f.write("\n")
    f.write("\n")
    print()

f.close()
