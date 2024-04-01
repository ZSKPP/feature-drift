import warnings
warnings.filterwarnings('ignore')

import numpy as np                                               #ver. 1.19.5
import pandas as pd                                              #ver. 1.0.0
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.kswin import KSWIN
from skmultiflow.drift_detection.page_hinkley import PageHinkley

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(arff.loadarff(fileName)[0])
    for i in range(0, len(dataFrame.columns)):
        col = classLabelEncoder.fit_transform(dataFrame.iloc[:, i].values)
        dataFrame.iloc[:, i] = col
    return dataFrame

def trainAndTest(classifier, dataFrame, position, trainingSamples):
    training = dataFrame.iloc[position: position + trainingSamples, :]
    X_train = training.drop(columns = "class")
    y_train = training["class"].astype('int')
    
    classifier.fit(X_train, y_train)
    
    tests = dataFrame.iloc[position + trainingSamples:, :]
    X = tests.drop(columns = "class")
    y = tests["class"].to_numpy().astype('int')

    y_predict = classifier.predict(X)

    return (y, y_predict)

def generateAccuracyWithoutDrifts(classifier, dataFrame, trainingSamples):
    accuracy = []
    (y, y_predict) = trainAndTest(classifier, dataFrame, 0, trainingSamples)
    # Assume: len(y) = len(y_predict)
    for i in range(0, len(y)):
        result = accuracy_score(y[0:(i + 1)], y_predict[0:(i + 1)])
        accuracy.append(result)
    
    return accuracy

def findDrift(y, y_predict, y_all, y_predict_all):
    # Assume: len(y) = len(y_predict)
    i = 0
    while i < len(y):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        driftDetector.add_element(y[i] == y_predict[i])
        i = i + 1
        if driftDetector.detected_change():
            driftDetector.reset()
            return i
    
    # -1 -> drift not detected
    return -1

def completeAccuracyArray(y, y_predict, y_all, y_predict_all, i, trainingSamples):
    k = 0
    while (k < trainingSamples) and (i < len(y)):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        i = i + 1
        k = k + 1

def generateAccuracyWithDrifts(classifier, driftDetector, dataFrame, trainingSamples):
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
            i = findDrift(y, y_predict, y_all, y_predict_all)
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
    for i in range(0, len(y_all)):
        result = accuracy_score(y_all[0:(i + 1)], y_predict_all[0:(i + 1)])
        accuracy.append(result)
    
    return (accuracy, drifts)  # (accuracyAll, accuracy)

##################################################################################
# Main()
TRAINING_SAMPLES = 500

inputPath = 'Benchmarks/'
fileName = "spam.arff"

classifier = RandomForestClassifier(random_state = 0)

# driftDetector = DDM()
driftDetector = EDDM() #ADWIN(delta = 0.002)
# driftDetector = HDDM_A(drift_confidence = 0.001, warning_confidence = 0.005, two_side_option = True)
# driftDetector = HDDM_W(drift_confidence = 0.001, warning_confidence = 0.005, lambda_option = 0.050, two_side_option = True)
# driftDetector = KSWIN(alpha = 0.005, window_size = 100, stat_size = 30, data = None)
# driftDetector = PageHinkley(min_instances = 30, delta = 0.005, threshold = 50, alpha = 1 - 0.0001)

dataFrame = loadArffData(inputPath + fileName)

(accuracyWithDrifts, drifts) = generateAccuracyWithDrifts(classifier, driftDetector, dataFrame, TRAINING_SAMPLES)
accuracyWithoutDrifts = generateAccuracyWithoutDrifts(classifier, dataFrame, TRAINING_SAMPLES)

print('Classifier name: %s' % classifier)
print('     Drift detector name: %s' % driftDetector.get_info())
print('     Benchmark name: %s' % fileName)
print('     Accuracy without drifts = %3.3f' % accuracyWithoutDrifts[-1])
print('     Accuracy with drifts and retraining classifier = %3.3f' % accuracyWithDrifts[-1])
print('     Number of drifts detected = %d' % len(drifts))

#print(len(accuracyWithoutDrifts))
#print(len(accuracyWithDrifts))

plt.subplot(2,1,1)
plt.ylim([0.0, 1.01])
plt.plot(range(0, len(accuracyWithoutDrifts)), accuracyWithoutDrifts, 'r')
plt.xlabel('Record', fontsize=12)
plt.ylabel('Accuracy (red)')
plt.tick_params(axis='both', which='major')
#plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
if len(drifts) > 0:
    plt.axvline(x = (drifts[0] - 1), color = 'b', linewidth=2)
    plt.axvline(x = (drifts[0] + TRAINING_SAMPLES - 1), color = 'g', linewidth = 2)
    for i in range(1, len(drifts)):
        plt.axvline(x = (drifts[i] - 1), color = 'b', linewidth=2)
        plt.axvline(x = (drifts[i] + TRAINING_SAMPLES - 1), color = 'g', linewidth = 2)

plt.ylim([0.0, 1.01])
plt.plot(range(0, len(accuracyWithDrifts)), accuracyWithDrifts, 'r')
plt.xlabel('Record', fontsize=12)
plt.ylabel('Accuracy (red)')
plt.tick_params(axis='both', which='major')
#plt.legend()
plt.grid(True)
plt.show()
