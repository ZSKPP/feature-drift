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

import warnings
warnings.filterwarnings("ignore")
import time
import os
import csv
import math
import numpy as np
import pandas as pd

from skmultiflow.trees import HoeffdingTreeClassifier

from scipy.io.arff import loadarff

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

from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)

class myHoeffdingTree:
    def __init__(self):
        self.classifier = HoeffdingTreeClassifier()

    def fit(self, X, y):
        self.classifier.fit(X.to_numpy(), y.to_numpy().astype("int"))

    def predict(self, X):
        return self.classifier.predict(X.to_numpy())

class FBDD():
    """ FBDD (Feature Based Drift Detector) method implementation."""

    def __init__(self, dataFrame, classifier, ranker, percentage_chunk_size, number_of_divisions, number_of_analyzed_features):
        self.__dataFrame = dataFrame
        self.__classifier = classifier
        self.__ranker = ranker
        self.__percentage_chunk_size = percentage_chunk_size
        self.__number_of_divisions = number_of_divisions
        self.__number_of_analyzed_features = number_of_analyzed_features

    def __setRanking(self, X, y):
        scores = []
        if self.__ranker == 'LASS':
            lasso = Lasso(alpha=0.05)
            scores.append(np.argsort(lasso.fit(X, y).coef_))
        K = 5
        if self.__ranker == 'LAPS':
            kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": K, 't': 1}
            W = construct_W.construct_W(X, **kwargs_W)
            scores.append(lap_score.lap_score(X, y, mode="index", W=W))

        return scores[0]

    def __setFeatureAndThreshold(self, chunk, records_in_chunk):
        numberOfChunks = int(chunk.shape[0] / records_in_chunk)
        split = np.array_split(chunk, numberOfChunks)
        results = []
        featuresNumber = len(chunk.columns) - 1
        places = [0] * featuresNumber
        for i in range(0, len(split)):
            X = split[i].drop(split[i].columns[len(split[i].columns) - 1], axis=1)
            y = split[i][split[i].columns[len(split[i].columns) - 1]]
            results.append(self.__setRanking(np.array(X), np.array(y)))
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

        featureRanking.sort(key=lambda x: x[1])

        return featureRanking

    def __detectDrifts(self, dataSplit, numberOfChunk, recordsInChunk, classCount, featureRanking, numberOfAnalyzedFeatures, debug):
        acc = []
        mcc = []
        prec = []
        recall = []  # RECALL = SENSITIVITY.
        f1 = []
        drifts = []

        for i in range(1, numberOfChunk):
            X = dataSplit[i].drop(dataSplit[i].columns[len(dataSplit[i].columns) - 1], axis=1)
            y = dataSplit[i][dataSplit[i].columns[len(dataSplit[i].columns) - 1]]
            y = y.astype('int')
            y_predict = self.__classifier.predict(X)

            acc.append(accuracy_score(y, y_predict))
            mcc.append(matthews_corrcoef(y, y_predict))

            if classCount > 2:
                prec.append(precision_score(y, y_predict, average='macro'))
                recall.append(recall_score(y, y_predict, average='macro'))
                f1.append(f1_score(y, y_predict, average='macro'))
            else:
                prec.append(precision_score(y, y_predict))
                recall.append(recall_score(y, y_predict))
                f1.append(f1_score(y, y_predict))

            score = self.__setRanking(np.array(X), np.array(y))

            if debug:
                print("**********************************************************************")
                print("Reference feature ranking:", featureRanking)
                print("Current feature ranking:", score)

            driftingFeatures = []
            driftOccurred = False
            for j in range(0, numberOfAnalyzedFeatures):
                feat = featureRanking[j][0]
                threshold = featureRanking[j][3]
                pos = np.where(score == feat)
                featurePos = pos[0][0]

                threshold1 = int(j - threshold)
                if threshold1 < 0:
                    threshold1 = 0
                threshold2 = int(j + threshold)
                if threshold2 > len(score) - 1:
                    threshold2 = len(score) - 1

                if debug:
                    print("Analyzing feature from reference ranking position %d (feature id %d): " % (j, feat))
                    print("\t Tolerable rank span is (%d, %d)" % (threshold1, threshold2))
                    print("\t Current rank is %d" % featurePos)

                if featurePos < threshold1:
                    driftingFeatures.append(j)
                    driftOccurred = True
                    break

                if featurePos > threshold2:
                    driftingFeatures.append(j)
                    driftOccurred = True
                    break

            if driftOccurred:
                if debug:
                    print("Drift on feature(s):", driftingFeatures)
                drifts.append(i - 1)
                # Re-learning
                chunk = i + 1
                if chunk < numberOfChunk:
                    X_train = dataSplit[chunk].drop(dataSplit[chunk].columns[len(dataSplit[chunk].columns) - 1], axis=1)
                    y_train = dataSplit[chunk][dataSplit[chunk].columns[len(dataSplit[chunk].columns) - 1]]
                    y_train = y_train.astype('int')
                    self.__classifier.fit(X_train, y_train)
                    featureRanking = self.__setFeatureAndThreshold(dataSplit[chunk], recordsInChunk / self.__number_of_divisions)

            if debug:
                print("**********************************************************************")

        return acc, mcc, f1, prec, recall, drifts

    def detectDrifts(self, debug):
        recordsInChunk = math.ceil(len(self.__dataFrame) * (self.__percentage_chunk_size / 100))
        numberOfChunk = int(np.ceil(self.__dataFrame.shape[0] / recordsInChunk))
        dataSplit = np.array_split(self.__dataFrame, numberOfChunk)

        classCount = len(self.__dataFrame[self.__dataFrame.columns[len(self.__dataFrame.columns) - 1]].unique())
        featureRanking = self.__setFeatureAndThreshold(dataSplit[0], recordsInChunk / self.__number_of_divisions)
        numberOfAnalyzedFeatures = self.__number_of_analyzed_features

        # Train RandomForest classifier by chunk number 0.
        X_train = dataSplit[0].drop(dataSplit[0].columns[len(dataSplit[0].columns) - 1], axis=1)
        y_train = dataSplit[0][dataSplit[0].columns[len(dataSplit[0].columns) - 1]]

        self.__classifier.fit(X_train, y_train)
        (acc, mcc, f1, prec, recall, drifts) = \
            self.__detectDrifts(dataSplit, numberOfChunk, recordsInChunk, classCount, featureRanking, numberOfAnalyzedFeatures, debug)

        return recordsInChunk, acc, np.mean(acc), np.mean(mcc), np.mean(f1), np.mean(prec), np.mean(recall), drifts

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

def DrawChart(driftsWithRetraining, driftsWithoutRetraining, accWithRetraining, accWithoutRetraining):
    plt.subplot(2, 1, 1)
    #if len(driftsWithoutRetraining) > 0:
    #    for i in range(0, len(driftsWithoutRetraining)):
    #        plt.axvline(x=(driftsWithoutRetraining[i]), color='b', linewidth=1)
    plt.plot(range(0, len(accWithoutRetraining)), accWithoutRetraining, 'r', linewidth=1)
    plt.xlabel('Instances', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.axis([-0.1, len(accWithoutRetraining), 0, 1.1])
    #plt.ylim([0,1.1])
    plt.grid(True)

    plt.subplot(2, 1, 2)
    if len(driftsWithRetraining) > 0:
        for i in range(0, len(driftsWithRetraining)):
            plt.axvline(x=(driftsWithRetraining[i]), color='b', linewidth=1)
            if (driftsWithRetraining[i]) <= len(driftsWithRetraining):
                plt.axvline(x=(driftsWithRetraining[i] + 1 - 0.03), color='g', linewidth=1)
    plt.plot(range(0, len(accWithRetraining)), accWithRetraining, 'r', linewidth=1)
    plt.xlabel('Instances', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.axis([-0.1, len(accWithRetraining), 0, 1.1])
    #plt.ylim([0,1.1])
    plt.grid(True)

    plt.show()

def FBDD_Detector(dataFrame, classifier, ranker, percentage_chunk_size, number_of_divisions, number_of_analyzed_features, debug):
    st = time.time()
    detector = FBDD(dataFrame, classifier, ranker, percentage_chunk_size, number_of_divisions, number_of_analyzed_features)
    (recordsInChunk, accWithRetraining, accValueWithRetraining, mccWithRetraining, f1WithRetraining, precWithRetraining, recallWithRetraining, driftsWithRetraining) = detector.detectDrifts(debug)
    en = time.time()
    timeWithRetraining = en - st

    st = time.time()
    detector = FBDD(dataFrame, classifier, ranker, percentage_chunk_size, number_of_divisions, 0)
    (recordsInChunk, accWithoutRetraining, accValueWithoutRetraining, mccWithoutRetraining, f1WithoutRetraining, precWithoutRetraining, recallWithoutRetraining, driftsWithoutRetraining) = detector.detectDrifts(debug)
    en = time.time()
    timeWithoutRetraining = en - st

    driftsInRawData = []
    for i in range(0, len(driftsWithRetraining)):
        driftsInRawData.append(driftsWithRetraining[i] * recordsInChunk)

    print("-----------------------------------------------------------")
    print(" Benchmark                      : ", fileName)
    print(" Classifier name                : ", type(classifier).__name__)
    print(" Drift detector name            :  FBDD")
    print(" Number of classes              : ", len(dataFrame[dataFrame.columns[len(dataFrame.columns) - 1]].unique()))
    print(" Records in chunk               : ", recordsInChunk)
    print(" Number of features             : ", dataFrame.shape[1] - 1)
    print("-----------------------------------------------------------")
    print("  Without drifts detection: ")
    print("     Accuracy                   : ", round(accValueWithoutRetraining, 3))
    print("     MCC                        : ", round(mccWithoutRetraining, 3))
    print("     Precision                  : ", round(precWithoutRetraining, 3))
    print("     Sensitivity (Recall)       : ", round(recallWithoutRetraining, 3))
    print("     F1                         : ", round(f1WithoutRetraining, 3))
    print("  Time                          : ", round(timeWithoutRetraining, 3))
    print("-----------------------------------------------------------")
    print("  With drifts detection and retraining: ")
    print("     Accuracy                   : ", round(accValueWithRetraining, 3))
    print("     MCC                        : ", round(mccWithRetraining, 3))
    print("     Precision                  : ", round(precWithRetraining, 3))
    print("     Sensitivity (Recall)       : ", round(recallWithRetraining, 3))
    print("     F1                         : ", round(f1WithRetraining, 3))
    print("  Time                          : ", round(timeWithRetraining, 3))
    print("-----------------------------------------------------------")
    print(" Number of drifts detected      : ", len(driftsWithRetraining))
    print(" Position of drifts in chunks   : ", driftsWithRetraining)
    print(" Position of drifts in raw data : ", driftsInRawData)
    print("-----------------------------------------------------------")

    DrawChart(driftsWithRetraining, driftsWithoutRetraining, accWithRetraining, accWithoutRetraining)

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

########################################################
classifier = HOEFF

inputPath = ""
fileName = "Abrupt_HP_1_10.arff"
dataFrame = loadData(inputPath + fileName)

FBDD_Detector(dataFrame=dataFrame,
              classifier=classifier,
              ranker="LASS",                # "LASS" or "LAPS"
              percentage_chunk_size=5,
              number_of_divisions=10,
              number_of_analyzed_features=1,
              debug=False)
