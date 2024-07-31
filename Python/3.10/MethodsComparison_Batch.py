# Python interpreter 3.10
# -----------------------
# Pandas               -> ver. 1.5.3
# Numpy                -> ver. 1.23.5
# Scipy                -> ver. 1.10.1
# Scikit-learn         -> ver. 1.2.2
# Frourous             -> ver. 0.6.1
# Xgboost              -> ver. 2.0.3
# Scikit-multiflow     -> ver. 0.5.3

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import math
import csv

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
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import trace_ratio

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

from frouros.metrics import PrequentialError

from xgboost import XGBClassifier

from skmultiflow.trees import HoeffdingTreeClassifier

class myHoeffdingTree:
    def __init__(self):
        self.classifier = HoeffdingTreeClassifier()
    def fit(self, X, y):
        self.classifier.fit(X.to_numpy(), y.to_numpy().astype('int'))
    def predict(self, X):
        return self.classifier.predict(X.to_numpy())

def loadData(fileName):
    # Load data from .arff or .csv file to DataFrame.
    if os.path.splitext(fileName)[1] == ".arff":
        dataFrame = pd.DataFrame(loadarff(fileName)[0])
    if os.path.splitext(fileName)[1] == ".csv":
        if not csv.Sniffer().has_header(open(fileName, 'r').read(4096)):
            dataFrame = pd.read_table(fileName, delimiter=str(csv.Sniffer().sniff(open(fileName, 'r').read()).delimiter), header=None)
        else:
            dataFrame = pd.read_table(fileName, delimiter=str(csv.Sniffer().sniff(open(fileName, 'r').read()).delimiter))
    # Encoding to numeric type.
    classLabelEncoder = LabelEncoder()
    for column in dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(dataFrame[column]):
            dataFrame[column] = classLabelEncoder.fit_transform(dataFrame[column])

    return dataFrame

def MakeClassSet(X, y, featuresNumber, instancesNumber):
    C1 = []
    C2 = []
    for i in range(0, featuresNumber):
        C1a = []
        C2a = []
        for j in range(0, instancesNumber):
            value = X[j][i]
            if y[j] == 1:
                C1a.append(value)
            else:
                C2a.append(value)
        C1.append(C1a)
        C2.append(C2a)

    return np.array(C1), np.array(C2)

def KolmogorovSmirnov(C1, C2, featuresNumber):
    ksFeatureValue = []
    for i in range(0, featuresNumber):
        ks = stats.ks_2samp(C1[i], C2[i])
        ksFeatureValue.append(ks.pvalue)

    return np.argsort(ksFeatureValue, 0)

def TTest(C1, C2, featuresNumber):
    ttFeatureValue = []
    for i in range(0, featuresNumber):
        tt = stats.ttest_ind(C1[i], C2[i], equal_var = True)
        ttFeatureValue.append(tt.pvalue)

    return np.argsort(ttFeatureValue, 0)

def CORR(X, C1, C2, featuresNumber):
    E1 = len(C1[0])
    E2 = len(C2[0])
    PK_1 = E1 / (E1 + E2)
    PK_2 = E2 / (E1 + E2)
    SR_f_mean = np.average(X)
    S_f = []
    for i in range(0, featuresNumber):
        SR_1 = np.average(C1[i])
        SR_2 = np.average(C2[i])
        C3 = np.concatenate((C1[i], C2[i]), axis = None)
        VAR_f = np.var(C3)
        A_f = PK_1 * (SR_1 - SR_f_mean) * (SR_1 - SR_f_mean) + PK_2 * (SR_2 - SR_f_mean) * (SR_2 - SR_f_mean)
        B_f = VAR_f * (PK_1 * (1 - PK_1) + PK_2 * (1 - PK_2))
        S_f.append(A_f / B_f)

    return np.argsort(S_f)[::-1]

def Ranking(X, y, ranker):
    scores = []
    if ranker == 'LASS':
        lasso = Lasso(alpha=0.001)
        scores.append(np.argsort(lasso.fit(X, y).coef_))

    if ranker == 'FISH':
        scores.append(fisher_score.fisher_score(X, y, mode="index"))
    '''
    if ranker == 'KS':
        featuresNumber = X.shape[1]
        instancesNumber = X.shape[0]
        C1, C2 = MakeClassSet(X, y, featuresNumber, instancesNumber)
        scores.append(KolmogorovSmirnov(C1, C2, featuresNumber))
    '''
    if ranker == 'TT':
        featuresNumber = X.shape[1]
        instancesNumber = X.shape[0]
        C1, C2 = MakeClassSet(X, y, featuresNumber, instancesNumber)
        scores.append(TTest(C1, C2, featuresNumber))
    '''
    if ranker == 'MUTI':
        aaa = MIC(X, y)
        scores.append(np.argsort(MIC(X, y))[::-1])     #  5. Mutual information.
    '''
    if ranker == 'CORR':
        featuresNumber = X.shape[1]
        instancesNumber = X.shape[0]
        C1, C2 = MakeClassSet(X, y, featuresNumber, instancesNumber)
        scores.append(CORR(X, C1, C2, featuresNumber))

    K = 5
    if ranker == 'LAPS':
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": K, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        scores.append(lap_score.lap_score(X, y, mode="index", W=W))

    if ranker == 'REL':
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": K, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        scores.append(reliefF.reliefF(X, y, mode="index", W=W))

    if ranker == 'TR':
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": K, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        scores.append(trace_ratio.trace_ratio(X, y, mode="index", W=W))

    if ranker == 'KW':
        fs = SelectKBest(f_classif, k='all')
        scores.append(np.argsort(fs.fit(X, y).scores_)[::-1])

    return scores[0]

def FBDD_computeSTDinChunk(chunk, records_in_chunk, ranker):
    numberOfChunks = int(chunk.shape[0] / records_in_chunk)
    split = np.array_split(chunk, numberOfChunks)
    results = []
    featuresNumber = len(chunk.columns) - 1
    places = [0] * featuresNumber
    for i in range(0, len(split)):
        X = split[i].drop(split[i].columns[len(split[i].columns)-1], axis=1)
        y = split[i][split[i].columns[len(split[i].columns)-1]]

        results.append(Ranking(np.array(X), np.array(y), ranker))

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
    feat = min(featureRanking, key=lambda t: t[1])

    return feat[0], feat[3]

def FBDD_detectDriftWithRetrainingClassifier(classifier, dataSplit, numberOfChunk, ranker, feat0, threshold, nod, classCount):
    acc = []
    mcc = []
    prec = []
    recall = []     # RECALL = SENSITIVITY.
    f1 = []

    drifts = []

    for i in range(1, numberOfChunk):
        X = dataSplit[i].drop(dataSplit[i].columns[len(dataSplit[i].columns)-1], axis=1)
        y = dataSplit[i][dataSplit[i].columns[len(dataSplit[i].columns)-1]]
        y = y.astype('int')
        y_predict = classifier.predict(X)

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

        score = Ranking(np.array(X), np.array(y), ranker)
        pos = np.where(score == feat0)
        if pos[0][0] > threshold:
            drifts.append(i - 1)
            # Re-learning
            chunk = i + 1
            if chunk < numberOfChunk:
                X_train = dataSplit[chunk].drop(dataSplit[chunk].columns[len(dataSplit[chunk].columns)-1], axis=1)
                y_train = dataSplit[chunk][dataSplit[chunk].columns[len(dataSplit[chunk].columns)-1]]
                y_train = y_train.astype('int')
                classifier.fit(X_train, y_train)
                (feat0, threshold) = FBDD_computeSTDinChunk(dataSplit[chunk], RECORDS_IN_CHUNK / nod, ranker)

#    return acc, mcc, prec, recall, f1, drifts
    return acc, f1, prec, drifts

def FBDD_WithRetraining(dataFrame, classifier, ranker, recordsInChunk, nod, classCount):
    numberOfChunk = int(np.ceil(dataFrame.shape[0] / recordsInChunk))
    dataSplit = np.array_split(dataFrame, numberOfChunk)
    (feat0, threshold) = FBDD_computeSTDinChunk(dataSplit[0], recordsInChunk / nod, ranker)

    # Train RandomForest classifier by chunk number 0.
    X_train = dataSplit[0].drop(dataSplit[0].columns[len(dataSplit[0].columns)-1], axis=1)
    y_train = dataSplit[0][dataSplit[0].columns[len(dataSplit[0].columns)-1]]

    classifier.fit(X_train, y_train)

    (acc, f1, prec, drifts) = \
        FBDD_detectDriftWithRetrainingClassifier(classifier, dataSplit, numberOfChunk, ranker, feat0, threshold, nod, classCount)

    return acc, f1, prec, drifts

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

def generateAccuracyWithDrifts(classifier, driftDetector, dataFrame, trainingSamples, classCount):
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

    #return accuracy, acc, mcc, prec, recall, f1, drifts
    return acc, f1, prec, drifts

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

XGB = XGBClassifier()

HOEFFDING = myHoeffdingTree()

classifiersArray = [
                        # [MLP,             "MLP"],
                        # [SVM_LINEAR,      "SVML"],
                        # [SVM_NON_LINEAR,  "SVMN"],
                        # [KNN,             "KNN"],
                        # [NAIVEBAYES,      "NB"],
                        [RF,              "RF"],
                        # [ADABOOST,        "ADA"],
                        # [GDA,             "GDA"],
                        # [EXTRATREES,      "ET"],
                        #[BAGGING,         "BAG"],
                        # [LDA,             "LDA"],
                        # [XGB,             "XGB"],
                        #[HOEFFDING,       "HOEF"],
                   ]

########################################################
##### Concept drift / Streaming / Change detection
########################################################

BOCDConfig = BOCDConfig()
BOCD = BOCD(config=BOCDConfig)
CUSUMConfig = CUSUMConfig()
CUSUM = CUSUM(config=CUSUMConfig)
GeometricMovingAverageConfig = GeometricMovingAverageConfig()
GeometricMovingAverage = GeometricMovingAverage(config=GeometricMovingAverageConfig)
PageHinkleyConfig = PageHinkleyConfig()
PageHinkley = PageHinkley(config=PageHinkleyConfig)

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

detectorsArray = [
                    # [BOCD,                   "BOCD"],
                    [CUSUM,                  "CUSUM"],
                    # [GeometricMovingAverage, "GMA"],
                    [PageHinkley,            "PH"],
                    [DDM,                    "DDM"],
                    #[ECDDWT,                 "ECDDWT"],
                    [EDDM,                   "EDDM"],
                    [HDDMA,                  "HDDM_A"],
                    [HDDMW,                  "HDDM_W"],
                    [RDDM,                   "RDDM"],
                    [ADWIN,                  "ADWIN"],
                    [KSWIN,                  "KSWIN"],
                    [STEPD,                  "STEPD"],
                  ]

########################################################
classifier = RF

#NUMBER_OF_DIVISIONS = 10
nod = 10
inputPath = ""
inputFileNames = [
    ["Abrupt_HP_1_10.arff"],
]
outputPath = "Results/"

outputFileName = "Results.txt"
f = open(outputPath + outputFileName, "w")

f.write("\\begin{table}[H]\n")
f.write("\\footnotesize\n")
f.write("\\centering\n")
f.write("\\captionsetup{justification=centering}\n")
f.write("\\begin{tabular}{|p{15mm}|")
for i in range(0, len(detectorsArray) + 2):
    f.write("C{15mm}|")
f.write("}\n")

for k in range(0, len(inputFileNames)):
    print(inputFileNames[k][0])
    #print("Detectors (%d) -> " % (len(detectorsArray) + 1), end="")
    #print("1 ", end="")

    f.write("\\hline \n")
    shortFileName = inputFileNames[k][0].replace(".arff","").replace(".csv","")
    f.write("\multicolumn{%d}{|c|}{Dataset: %s} \\\ \\hline\n" % (len(detectorsArray)+3, shortFileName))
    line = "Metrics & FBDD_1 & FBDD_2"
    for i in range(0, len(detectorsArray)):
        line += " & " + detectorsArray[i][1]
    f.write(line)
    f.write("\\\ \\hline \n")

    dataFrame = loadData(inputPath + inputFileNames[k][0])
    RECORDS_IN_CHUNK = math.ceil(len(dataFrame) * 0.05)
    NUMBER_OF_CLASSES = len(dataFrame[dataFrame.columns[len(dataFrame.columns) - 1]].unique())

    accArray = []
    f1Array = []
    precArray = []
    driftsArray = []

    print("1 ", end="")
    (acc, f1, prec, drifts) = FBDD_WithRetraining(dataFrame, classifier, "LASS", RECORDS_IN_CHUNK, nod, NUMBER_OF_CLASSES)
    accArray.append(np.mean(acc))
    f1Array.append(np.mean(f1))
    precArray.append(np.mean(prec))
    driftsArray.append(len(drifts))

    print("2 ", end="")
    (acc, f1, prec, drifts) = FBDD_WithRetraining(dataFrame, classifier, "LAPS", RECORDS_IN_CHUNK, nod, NUMBER_OF_CLASSES)
    accArray.append(np.mean(acc))
    f1Array.append(np.mean(f1))
    precArray.append(np.mean(prec))
    driftsArray.append(len(drifts))

    for j in range(0, len(detectorsArray)):
        print('%d ' % (j + 3), end="")
        driftDetector = detectorsArray[j][0]
        driftDetector.reset()
        (acc, f1, prec, drifts) = generateAccuracyWithDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK, NUMBER_OF_CLASSES)
        accArray.append(np.mean(acc))
        f1Array.append(np.mean(f1))
        precArray.append(np.mean(prec))
        driftsArray.append(len(drifts))

    f.write("ACC")
    for j in range(0, len(accArray)):
        f.write(" & %3.3f/%d " % (accArray[j], driftsArray[j]))
    f.write("\\\ \\hline \n")

    f.write("F1")
    for j in range(0, len(f1Array)):
        f.write(" & %3.3f " % (f1Array[j]))
    f.write("\\\ \\hline \n")

    f.write("Precision")
    for j in range(0, len(precArray)):
        f.write(" & %3.3f " % (precArray[j]))
    f.write("\\\ \\hline \n")

    print()

f.write("\\end{tabular}\n")
f.write("\\caption{Average metrics for RF classifier and drift detectors/number of detected drifts. Data with unknown drift points.}\n")
f.write("\\label{Tabela}\n")
f.write("\\end{table}\n")

f.close()
