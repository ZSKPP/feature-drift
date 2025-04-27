import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import math
import csv
import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)

from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from xgboost import XGBClassifier
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W

class FBDD():
    def __init__(self, dataFrame, classifier, ranker, percentage_chunk_size, number_of_divisions,
                 number_of_analyzed_features):
        self.__dataFrame = dataFrame
        self.__classifier = classifier
        self.__ranker = ranker
        self.__percentage_chunk_size = percentage_chunk_size
        self.__number_of_divisions = number_of_divisions
        self.__number_of_analyzed_features = number_of_analyzed_features

    def __setRanking(self, X, y):
        scores = []
        if self.__ranker == 'LASS':
            lasso = Lasso(alpha=0.001)
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
        for split_chunk in split:
            X = split_chunk.drop(split_chunk.columns[-1], axis=1)
            y = split_chunk[split_chunk.columns[-1]]
            results.append(self.__setRanking(np.array(X), np.array(y)))

        featureRanking = []
        for j in range(featuresNumber):
            featPlace = [np.where(r == j)[0][0] for r in results]
            threshold = np.mean(featPlace) + np.std(featPlace)
            threshold = np.ceil(threshold)
            featureRanking.append((j, np.mean(featPlace), np.std(featPlace), threshold))
        featureRanking.sort(key=lambda x: x[1])
        return featureRanking

    def __detectDriftsWithoutRetraining(self, dataSplit, numberOfChunk, classCount, featureRanking,
                                        numberOfAnalyzedFeatures):
        acc, mcc, prec, f1, drifts = [], [], [], [], []
        for i in range(1, numberOfChunk):
            X = dataSplit[i].drop(dataSplit[i].columns[-1], axis=1)
            y = dataSplit[i][dataSplit[i].columns[-1]].astype('int')
            y_predict = self.__classifier.predict(X)
            acc.append(accuracy_score(y, y_predict))
            mcc.append(matthews_corrcoef(y, y_predict))
            avg = 'macro' if classCount > 2 else 'binary'
            prec.append(precision_score(y, y_predict, average=avg, zero_division=0))
            f1.append(f1_score(y, y_predict, average=avg, zero_division=0))
            score = self.__setRanking(np.array(X), np.array(y))
            for j in range(numberOfAnalyzedFeatures):
                feat, threshold = featureRanking[j][0], featureRanking[j][3]
                featurePos = np.where(score == feat)[0][0]
                if featurePos < max(0, j - threshold) or featurePos > min(len(score) - 1, j + threshold):
                    drifts.append(i - 1)
                    break
        return acc, f1, prec, drifts, mcc

    def __detectDriftsWithRetraining(self, dataSplit, numberOfChunk, recordsInChunk, classCount, featureRanking,
                                     numberOfAnalyzedFeatures):
        acc, mcc, prec, f1, drifts = [], [], [], [], []
        for i in range(1, numberOfChunk):
            X = dataSplit[i].drop(dataSplit[i].columns[-1], axis=1)
            y = dataSplit[i][dataSplit[i].columns[-1]].astype('int')
            y_predict = self.__classifier.predict(X)
            acc.append(accuracy_score(y, y_predict))
            mcc.append(matthews_corrcoef(y, y_predict))
            avg = 'macro' if classCount > 2 else 'binary'
            prec.append(precision_score(y, y_predict, average=avg, zero_division=0))
            f1.append(f1_score(y, y_predict, average=avg, zero_division=0))
            score = self.__setRanking(np.array(X), np.array(y))
            driftOccurred = False
            for j in range(numberOfAnalyzedFeatures):
                feat, threshold = featureRanking[j][0], featureRanking[j][3]
                featurePos = np.where(score == feat)[0][0]
                if featurePos < max(0, j - threshold) or featurePos > min(len(score) - 1, j + threshold):
                    drifts.append(i - 1)
                    driftOccurred = True
                    break
            if driftOccurred and (i + 1) < numberOfChunk:
                X_train = dataSplit[i + 1].drop(dataSplit[i + 1].columns[-1], axis=1)
                y_train = dataSplit[i + 1][dataSplit[i + 1].columns[-1]].astype('int')
                self.__classifier.fit(X_train, y_train)
                featureRanking = self.__setFeatureAndThreshold(dataSplit[i + 1],
                                                              recordsInChunk / self.__number_of_divisions)
        return acc, f1, prec, drifts, mcc

    def detectDrifts(self):
        recordsInChunk = math.ceil(len(self.__dataFrame) * (self.__percentage_chunk_size / 100))
        numberOfChunk = int(np.ceil(self.__dataFrame.shape[0] / recordsInChunk))
        dataSplit = np.array_split(self.__dataFrame, numberOfChunk)
        classCount = len(self.__dataFrame[self.__dataFrame.columns[-1]].unique())
        X_train = dataSplit[0].drop(dataSplit[0].columns[-1], axis=1)
        y_train = dataSplit[0][dataSplit[0].columns[-1]].astype('int')
        self.__classifier.fit(X_train, y_train)
        featureRanking = self.__setFeatureAndThreshold(dataSplit[0], recordsInChunk / self.__number_of_divisions)
        numberOfAnalyzedFeatures = self.__number_of_analyzed_features
        acc_without, f1_without, prec_without, drifts_without, mcc_without = self.__detectDriftsWithoutRetraining(
            dataSplit, numberOfChunk, classCount, featureRanking, numberOfAnalyzedFeatures)
        self.__classifier.fit(X_train, y_train)
        featureRanking = self.__setFeatureAndThreshold(dataSplit[0], recordsInChunk / self.__number_of_divisions)
        acc_with, f1_with, prec_with, drifts_with, mcc_with = self.__detectDriftsWithRetraining(
            dataSplit, numberOfChunk, recordsInChunk, classCount, featureRanking, numberOfAnalyzedFeatures)
        return (np.mean(acc_without), np.mean(f1_without), np.mean(prec_without), np.mean(mcc_without), drifts_without,
                np.mean(acc_with), np.mean(f1_with), np.mean(prec_with), np.mean(mcc_with), drifts_with, recordsInChunk,
                acc_without, acc_with)

def loadData(fileName):
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
    classLabelEncoder = LabelEncoder()
    for column in _dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(_dataFrame[column]):
            _dataFrame[column] = classLabelEncoder.fit_transform(_dataFrame[column])
    return _dataFrame

def get_classifiers():
    return {
        "RF": RandomForestClassifier(random_state=31),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(probability=True, random_state=31),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "Perceptron": Perceptron(),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

def DrawChart(driftsWithRetraining, driftsWithoutRetraining, accWithRetraining, accWithoutRetraining):
    plt.subplot(2, 1, 1)
    plt.plot(range(0, len(accWithoutRetraining)), accWithoutRetraining, 'r', linewidth=1)
    plt.xlabel('Instances', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.axis([-0.1, len(accWithoutRetraining), 0, 1.1])
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
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main ---
inputPath = "Benchmarks/"
fileName = "outdoor.arff"
dataFrame = loadData(inputPath + fileName)
classifiers = get_classifiers()
clf_name = "RF"
classifier = classifiers[clf_name]

fbdd = FBDD(dataFrame=dataFrame, classifier=classifier, ranker="LASS", percentage_chunk_size=5, number_of_divisions=10,
            number_of_analyzed_features=1)

(acc_without, f1_without, prec_without, mcc_without, drifts_without,
 acc_with, f1_with, prec_with, mcc_with, drifts_with, recordsInChunk,
 acc_without_list, acc_with_list) = fbdd.detectDrifts()

print("=== Without retraining ===")
print("Records in chunk :", recordsInChunk)
print(f"Accuracy   = {acc_without:.3f}")
print(f"MCC coeff. = {mcc_without:.3f}")
print(f"F1 score   = {f1_without:.3f}")
print(f"Precision  = {prec_without:.3f}")
print("Number of drifts:", len(drifts_without))
print(drifts_without)

print("\n=== With retraining ===")
print("Records in chunk :", recordsInChunk)
print(f"Accuracy   = {acc_with:.3f}")
print(f"MCC coeff. = {mcc_with:.3f}")
print(f"F1 score   = {f1_with:.3f}")
print(f"Precision  = {prec_with:.3f}")
print("Number of drifts:", len(drifts_with))
print(drifts_with)

DrawChart(drifts_with, drifts_without, acc_with_list, acc_without_list)