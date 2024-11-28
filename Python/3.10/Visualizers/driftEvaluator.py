import math
import os
import csv

import numpy
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frouros.metrics import PrequentialError
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

# Concept drift / Streaming / Statistical process control
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.concept_drift import EDDM, EDDMConfig
from frouros.detectors.concept_drift import ECDDWT, ECDDWTConfig
from frouros.detectors.concept_drift import HDDMA, HDDMAConfig
from frouros.detectors.concept_drift import HDDMW, HDDMWConfig
from frouros.detectors.concept_drift import RDDM, RDDMConfig

from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.concept_drift import KSWIN, KSWINConfig
from frouros.detectors.concept_drift import STEPD, STEPDConfig

##############################################################

# Read input file
fileName = "benchmarks/phishing.arff"

if os.path.splitext(fileName)[1] == ".arff":
    df_data = pd.DataFrame(loadarff(fileName)[0])
if os.path.splitext(fileName)[1] == ".csv":
    if not csv.Sniffer().has_header(open(fileName, "r").read(2048)):
        df_data = pd.read_table(
            fileName,
            delimiter=str(csv.Sniffer().sniff(open(fileName, "r").read()).delimiter),
            header=None,
        )
    else:
        df_data = pd.read_table(
            fileName,
            delimiter=str(csv.Sniffer().sniff(open(fileName, "r").read()).delimiter),
        )

# Encode class labels as integers
classLabelEncoder = LabelEncoder()
for column in df_data.columns:
    if not pd.api.types.is_numeric_dtype(df_data[column]):
        df_data[column] = classLabelEncoder.fit_transform(df_data[column])

classCount = len(df_data[df_data.columns[len(df_data.columns) - 1]].unique())

# Split into data and labels
X_data = df_data.iloc[:, :-1].to_numpy()
Y_labels = df_data.iloc[:, -1].to_numpy().astype("int")

# Training sets for classifier and detector
detectorTrainingSize = 30  # Number of instances used to train the drift detector
classifierTrainingSize = 850  # Number of instances used to train the classifier
# classifierTrainingSize = math.ceil(len(df_data) * 0.055)

# Classifier pipeline definition
classifierModel = Pipeline(
    [
        # ("scaler", StandardScaler()),
        # ("model", DecisionTreeClassifier()),
        ("model", GaussianNB()),
        # ("model", RandomForestClassifier(random_state=31)),
    ]
 )

# classifierModel = GaussianNB()

# Detectors for Concept drift / Streaming / Statistical process control
detectors = [
    # ('DDM', DDM(config=DDMConfig(min_num_instances=detectorTrainingSize))),
    # ('EDDM', EDDM(config=EDDMConfig(min_num_misclassified_instances=detectorTrainingSize))),
    # ('ECDDWT', ECDDWT(config=ECDDWTConfig(min_num_instances=detectorTrainingSize))),
    ("HDDMA", HDDMA(config=HDDMAConfig())),
    # ('HDDMW', HDDMW(config=HDDMWConfig(min_num_instances=detectorTrainingSize))),
    # ('RDDM', RDDM(config=RDDMConfig())),
    # ('ADWIN', ADWIN(config=ADWINConfig())),
    # ('STEPD', STEPD(config=STEPDConfig())),
    # ('KSWIN', KSWIN(config=KSWINConfig())),
]

trueDriftIdx = []  # 50000 - classifierTrainingSize]

# Main loop for testing detectors
for dname, detector in detectors:
    print("Detector:", dname)
    # Metrics and initializations
    mtrPreqErr = PrequentialError(alpha=0.999)
    predictions = []
    errorValues, driftDetectIdx, modelSwitchIdx, detectorBeginIdx = [], [], [], []
    newModelIndicator = False
    detector.reset()
    mtrPreqErr.reset()
    driftDetected = False
    collectedSamples = 0
    lastDetectorResetIdx = 0

    # Split initial train and test, take from top, NO SHUFFLING!!!
    (X_train, X_test, Y_train, Y_test) = train_test_split(
        X_data, Y_labels, train_size=classifierTrainingSize, shuffle=False
    )

    # training = df_data.iloc[0:classifierTrainingSize, :]
    # X_train = training.iloc[:, :-1].to_numpy()
    # Y_train = training.iloc[:, -1].to_numpy().astype("int")

    # tests = df_data.iloc[classifierTrainingSize:, :]
    # X_test = tests.iloc[:, :-1].to_numpy()
    # Y_test = tests.iloc[:, -1].to_numpy().astype("int")

    # Create initial classification model
    classifierModel.fit(X_train, Y_train)

    # Simulate data stream (assuming test label available after prediction)
    for i in tqdm(range(X_test.shape[0])):
        x = X_test[i]
        y = Y_test[i]

        Y_pred = classifierModel.predict(x.reshape(1, -1)).item()

        predictions.append(Y_pred)
        errorMetric = 1 - int(Y_pred == y)
        errorValues.append(mtrPreqErr(error_value=errorMetric))
        detector.update(value=errorMetric)

        if detector.drift:
            driftDetected = True

        if driftDetected:
            if collectedSamples == 0:  # First data vector after new drift was detected
                X_train = []  # initialize a training set for a new model
                Y_train = []
                driftDetectIdx.append(i)  # Mark drift detection
                print(" - drift detected @", i)
            X_train.append(x)
            Y_train.append(y)
            collectedSamples += 1
            if collectedSamples == classifierTrainingSize:
                classifierModel.fit(
                    X_train, Y_train
                )  # Create/substitute a new classification model
                #detector = HDDMA(config=HDDMAConfig())
                detector.reset()  # reset detector, from now have to wait to collect detectorTrainingSize again !!!DOES NOT WORK WITH HDDMA!!!
                mtrPreqErr.reset()
                collectedSamples = 0
                lastDetectorResetIdx = i
                driftDetected = False
                modelSwitchIdx.append(i)  # Mark classification model switching

        if i - lastDetectorResetIdx == detectorTrainingSize:
            detectorBeginIdx.append(i)

    acc = accuracy_score(Y_test, predictions)
    mcc = matthews_corrcoef(Y_test, predictions)
    if classCount > 2:
        prec = precision_score(Y_test, predictions, average="macro")
        recall = recall_score(Y_test, predictions, average="macro")
        f1 = f1_score(Y_test, predictions, average="macro")
    else:
        prec = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)
    print("Number of detected drifts:", len(driftDetectIdx))
    print("Indices where drift occurred:", driftDetectIdx)
    print("     ACC: ", round(acc, 3))
    print("     MCC: ", round(mcc, 3))
    print("     PRECISION: ", round(prec, 3))
    print("     SENSITIVITY: ", round(recall, 3))
    print("     F1: ", round(f1, 3))
    print()

    # numpy.savetxt("predVec.csv", predictions, delimiter=";", fmt="%d")

    # Plot prequential error and drift detection points
    plt.figure(figsize=(14, 2))
    plt.ylim(0, 1)
    plt.plot(errorValues, label="Prequential Error")
    plt.eventplot(
        trueDriftIdx,
        orientation="horizontal",
        lineoffsets=0.5,
        colors="m",
        linestyles="-",
        label="Real drift posiotion",
    )
    plt.eventplot(
        driftDetectIdx,
        orientation="horizontal",
        lineoffsets=0.5,
        colors="r",
        linestyles="--",
        label="Drift detected",
    )
    plt.eventplot(
        modelSwitchIdx,
        orientation="horizontal",
        lineoffsets=0.5,
        colors="g",
        linestyles="-.",
        label="Model switched",
    )
    plt.eventplot(
        detectorBeginIdx,
        orientation="horizontal",
        lineoffsets=0.5,
        colors="b",
        linestyles=":",
        label="Start looking for a drift",
    )
    # for idx in driftDetectIdx:
    #    plt.axvline(x=idx, color='r', linestyle='--', label="Drift detected")
    plt.xlabel("Instances")
    plt.ylabel("Error Rate")
    plt.title("Detector: " + dname)
    plt.legend()
    plt.show()
