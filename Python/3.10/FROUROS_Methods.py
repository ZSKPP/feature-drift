# Python interpreter 3.8
# -----------------------
# Pandas               -> ver. 1.5.3
# Numpy                -> ver. 1.23.5
# Scipy                -> ver. 1.10.1
# Scikit-learn         -> ver. 1.2.2
# MatplotLib           -> ver. 3.7.5
# Xgboost              -> ver. 2.0.3
# Scikit-multiflow     -> ver. 0.5.3
# Skfeature-chappers   -> ver. 1.1.0
# Frouros              -> ver. 0.6.1

import warnings
warnings.filterwarnings("ignore")
import time
import os
import csv
import math
import numpy as np
import pandas as pd

from skmultiflow.trees import HoeffdingTreeClassifier

from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, Perceptron, LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier

from xgboost import XGBClassifier
from scipy.io.arff import loadarff

#####      frouros #ver. 0.6.1
# Concept drift / Streaming / Change detection
from frouros.detectors.concept_drift import BOCD, BOCDConfig
from frouros.detectors.concept_drift import CUSUM, CUSUMConfig
from frouros.detectors.concept_drift import (
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
)
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
from frouros.metrics import PrequentialError
import matplotlib.pyplot as plt

class myHoeffdingTree:
    def __init__(self):
        self.classifier = HoeffdingTreeClassifier()

    def fit(self, X, y):
        self.classifier.fit(X.to_numpy(), y.to_numpy().astype("int"))

    def predict(self, X):
        return self.classifier.predict(X.to_numpy())

def loadData(fileName):
    # Load data from .arff or .csv file to DataFrame.
    if os.path.splitext(fileName)[1] == ".arff":
        _dataFrame = pd.DataFrame(loadarff(fileName)[0])
    if os.path.splitext(fileName)[1] == ".csv":
        if not csv.Sniffer().has_header(open(fileName, 'r').read(4096)):
            _dataFrame = pd.read_table(fileName,
                                       delimiter=str(csv.Sniffer().sniff(open(fileName, 'r').read()).delimiter),
                                       header=None)
        else:
            _dataFrame = pd.read_table(fileName,
                                       delimiter=str(csv.Sniffer().sniff(open(fileName, 'r').read()).delimiter))
    # Encoding to numeric type.
    classLabelEncoder = LabelEncoder()
    for column in _dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(_dataFrame[column]):
            _dataFrame[column] = classLabelEncoder.fit_transform(_dataFrame[column])

    return _dataFrame

def trainAndTest(classifier, dataFrame, position, trainingSamples):
    training = dataFrame.iloc[position : position + trainingSamples, :]
    X_train = training.drop(training.columns[len(training.columns) - 1], axis=1)
    y_train = training[training.columns[len(training.columns) - 1]].astype("int")

    classifier.fit(X_train, y_train)

    tests = dataFrame.iloc[position + trainingSamples :, :]
    X = tests.drop(tests.columns[len(tests.columns) - 1], axis=1)
    y = tests[tests.columns[len(tests.columns) - 1]].to_numpy().astype("int")

    y_predict = classifier.predict(X)

    return y, y_predict

def generateAccuracyWithoutDrifts(classifier, dataFrame, trainingSamples, classCount):
    metric = PrequentialError(alpha=0.999)
    accuracyWithoutDrifts = []
    (y, y_predict) = trainAndTest(classifier, dataFrame, 0, trainingSamples)
    # Assume: len(y) = len(y_predict)
    for i in range(0, len(y)):
        if y[i : (i + 1)] == y_predict[i : (i + 1)]:
            error = 0
        else:
            error = 1
        metric_error = metric(error_value=error)
        # accuracyWithoutDrifts.append(metric_error)
        accuracyWithoutDrifts.append(1 - metric_error)

    acc = accuracy_score(y, y_predict)
    mcc = matthews_corrcoef(y, y_predict)

    if classCount > 2:
        prec = precision_score(y, y_predict, average="macro")
        recall = recall_score(y, y_predict, average="macro")
        f1 = f1_score(y, y_predict, average="macro")
    else:
        prec = precision_score(y, y_predict)
        recall = recall_score(y, y_predict)
        f1 = f1_score(y, y_predict)

    return accuracyWithoutDrifts, acc, mcc, prec, recall, f1

def findDrift(y, y_predict, y_all, y_predict_all, driftDetector):
    # Assume: len(y) = len(y_predict)
    # metric = PrequentialError(alpha = 0.999)
    i = 0
    while i < len(y):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        if y[i : (i + 1)] == y_predict[i : (i + 1)]:
            error = 0
        else:
            error = 1
        # metric_error = metric(error_value=error)

        _ = driftDetector.update(value=error)
        if driftDetector.drift:
            driftDetector.reset()
            return i
        i = i + 1

    return -1

def completeAccuracyArray(y, y_predict, y_all, y_predict_all, i, trainingSamples):
    k = 0
    while (k < trainingSamples) and (i < len(y)):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        i = i + 1
        k = k + 1


def generateAccuracyWithDrifts(
    classifier, driftDetector, dataFrame, trainingSamples, classCount
):
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
            if i == -1:
                finish = True
            else:
                drft = drft + i + t
                drifts.append(drft)
                j = drft + trainingSamples
                completeAccuracyArray(
                    y, y_predict, y_all, y_predict_all, i, trainingSamples
                )
            t = trainingSamples
        else:
            finish = True

    accuracy = []
    metric = PrequentialError(alpha=0.999)
    # Assume: len(y_all) = len(y_predict_all)
    for i in range(0, len(y_all)):
        if y_all[i : (i + 1)] == y_predict_all[i : (i + 1)]:
            error = 0
        else:
            error = 1
        metric_error = metric(error_value=error)
        # accuracy.append(metric_error)
        accuracy.append(1 - metric_error)

    acc = accuracy_score(y_all, y_predict_all)
    mcc = matthews_corrcoef(y_all, y_predict_all)

    if classCount > 2:
        prec = precision_score(y_all, y_predict_all, average="macro")
        recall = recall_score(y_all, y_predict_all, average="macro")
        f1 = f1_score(y_all, y_predict_all, average="macro")
    else:
        prec = precision_score(y_all, y_predict_all)
        recall = recall_score(y_all, y_predict_all)
        f1 = f1_score(y_all, y_predict_all)

    return accuracy, acc, mcc, prec, recall, f1, drifts

def DrawChart(accuracyWithoutDrifts, accuracyWithDrifts, drifts):
    plt.subplot(2, 1, 1)
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithoutDrifts)), accuracyWithoutDrifts, "r")
    plt.xlabel("Record", fontsize=12)
    plt.ylabel("Error rate")
    plt.tick_params(axis="both", which="major")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    if len(drifts) > 0:
        plt.axvline(x=(drifts[0] - 1), color="b", linewidth=2)
        if (drifts[0] + RECORDS_IN_CHUNK - 1) < len(accuracyWithDrifts):
            plt.axvline(x=(drifts[0] + RECORDS_IN_CHUNK - 1), color="g", linewidth=2)
        for i in range(1, len(drifts)):
            plt.axvline(x=(drifts[i] - 1), color="b", linewidth=2)
            if (drifts[i] + RECORDS_IN_CHUNK - 1) < len(accuracyWithDrifts):
                plt.axvline(
                    x=(drifts[i] + RECORDS_IN_CHUNK - 1), color="g", linewidth=2
                )
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithDrifts)), accuracyWithDrifts, "r")
    plt.xlabel("Record", fontsize=12)
    plt.ylabel("Error rate")
    plt.tick_params(axis="both", which="major")
    plt.grid(True)
    plt.show()

##################################################################################
# Main()
##################################################################################

########################################################
# Classifiers.
########################################################
np.random.seed(seed=31)
random_state = 31

PPN = Perceptron(max_iter=1000, random_state=random_state)
LR = LogisticRegression(solver='liblinear', random_state=random_state)
MLP = MLPClassifier(max_iter=1000, random_state=random_state)

SVM_LINEAR = SVC(kernel='linear', random_state=random_state)
SVM_NON_LINEAR = SVC(kernel='rbf', gamma='auto', random_state=random_state)
SVM_POLY = SVC(kernel='poly', gamma='auto', degree=1, random_state=random_state)

TREE = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=random_state)
KNN = KNeighborsClassifier(n_neighbors=10, metric='euclidean', weights='distance')
NAIVEBAYES = GaussianNB()
SGD = SGDClassifier(max_iter=1000, random_state=random_state)
GPC = GaussianProcessClassifier(random_state=random_state)

RF = RandomForestClassifier(random_state=random_state)
ADABOOST = AdaBoostClassifier(random_state=random_state)
GDA = GradientBoostingClassifier(random_state=random_state)
EXTRATREES = ExtraTreesClassifier(random_state=random_state)
BAGGING = BaggingClassifier(TREE, random_state=random_state)

LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()

HOEFF = myHoeffdingTree()
XGB = XGBClassifier()

####################################
# Concept drift detection methods. #
####################################

##### Concept drift / Streaming / Change detection
BOCD = BOCD(config=BOCDConfig())
CUSUM = CUSUM(config=CUSUMConfig())
GMA = GeometricMovingAverage(config=GeometricMovingAverageConfig())
PH = PageHinkley(config=PageHinkleyConfig())

##### Concept drift / Streaming / Statistical process control
DDM = DDM(config=DDMConfig())
ECDDWT = ECDDWT(config=ECDDWTConfig())
EDDM = EDDM(config=EDDMConfig())
HDDMA = HDDMA(config=HDDMAConfig())
HDDMW = HDDMW(config=HDDMWConfig())
RDDM = RDDM(config=RDDMConfig())

##### Concept drift / Streaming / Window based
ADWIN = ADWIN(config=ADWINConfig())
KSWIN = KSWIN(config=KSWINConfig())
STEPD = STEPD(config=STEPDConfig())

########################################################

classifier = HOEFF
driftDetector = ADWIN

inputPath = ""
fileName = "Abrupt_HP_1_10.arff"

dataFrame = loadData(inputPath + fileName)
RECORDS_IN_CHUNK = math.ceil(len(dataFrame) * 0.05)

NUMBER_OF_CLASSES = len(dataFrame[dataFrame.columns[len(dataFrame.columns) - 1]].unique())

st = time.time()
(accuracyWithoutDrifts, accWithoutDrifts, mccWithoutDrifts, precWithoutDrifts, recallWithoutDrifts, f1WithoutDrifts) =\
    generateAccuracyWithoutDrifts(classifier, dataFrame, RECORDS_IN_CHUNK, NUMBER_OF_CLASSES)
en = time.time()
timeWithoutDrifts = en - st

st = time.time()
(accuracyWithDrifts, accWithDrifts, mccWithDrifts, precWithDrifts, recallWithDrifts, f1WithDrifts, drifts) =\
    generateAccuracyWithDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK, NUMBER_OF_CLASSES)
en = time.time()
timeWithDrifts = en - st

print("-----------------------------------------------------------")
print(" Benchmark                 : ", fileName)
print(" Classifier name           : ", type(classifier).__name__)
print(" Drift detector name       : ", type(driftDetector).__name__)
print(" Number of class           : ", NUMBER_OF_CLASSES)
print(" Records in chunk          : ", RECORDS_IN_CHUNK)
print(" Number of features        : ", dataFrame.shape[1] - 1)
print("-----------------------------------------------------------")
print("  Without drifts detection: ")
print("     Accuracy                   : ", round(accWithoutDrifts, 3))
print("     MCC                        : ", round(mccWithoutDrifts, 3))
print("     Precision                  : ", round(precWithoutDrifts, 3))
print("     Sensitivity (Recall)       : ", round(recallWithoutDrifts, 3))
print("     F1                         : ", round(f1WithoutDrifts, 3))
print("  Time                          : ", round(timeWithoutDrifts, 3))
print("-----------------------------------------------------------")
print("  With drifts detection and retraining: ")
print("     Accuracy                   : ", round(accWithDrifts, 3))
print("     MCC                        : ", round(mccWithDrifts, 3))
print("     Precision                  : ", round(precWithDrifts, 3))
print("     Sensitivity (Recall)       : ", round(recallWithDrifts, 3))
print("     F1                         : ", round(f1WithDrifts, 3))
print("  Time                          : ", round(timeWithDrifts, 3))
print("-----------------------------------------------------------")
print(" Number of drifts detected      : ", len(drifts))
print(" Position of drifts in raw data : ", drifts)
print("-----------------------------------------------------------")

DrawChart(accuracyWithoutDrifts, accuracyWithDrifts, drifts)
