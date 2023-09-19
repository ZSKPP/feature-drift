import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd                                         #ver. 1.5.3
import numpy as np                                          #ver. 1.23.5
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.kswin import KSWIN
from skmultiflow.drift_detection.page_hinkley import PageHinkley

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(loadarff(fileName)[0])
    i = len(dataFrame.columns)-1
    col = classLabelEncoder.fit_transform(dataFrame.iloc[:, i].values)
    dataFrame.isetitem(i, col)
    return dataFrame

def generateAccuracy(classifier, dataFrame, recordsInChunk):
    numberOfChunk = int(dataFrame.shape[0] / recordsInChunk)
    dataSplit = np.array_split(dataFrame, numberOfChunk)
    X_train = dataSplit[0].drop(columns = "class")
    y_train = dataSplit[0]["class"]
    classifier.fit(X_train, y_train)
    accuracy = []
    for i in range(1, numberOfChunk):
        X = dataSplit[i].drop(columns = "class")
        y = dataSplit[i]["class"]
        y_predict = classifier.predict(X)
        result = accuracy_score(y, y_predict)
        accuracy.append(result)
    return accuracy

def generateDrifts(classifier, driftDetector, dataFrame, recordsInChunk):
    training = dataFrame.iloc[:recordsInChunk, :]
    tests = dataFrame.iloc[recordsInChunk:, :]
    X_train = training.drop(training.columns[[len(dataFrame.columns) - 1]], axis = 1)
    y_train = training.iloc[:, len(dataFrame.columns) - 1]
    classifier.fit(X_train, y_train)
    X_test = tests.drop(tests.columns[[len(dataFrame.columns) - 1]], axis = 1)
    y_test = tests.iloc[:, len(dataFrame.columns) - 1]
    y_predict = classifier.predict(X_test)
    y_test = np.array(y_test)
    drifts = []
    for i in range(0, len(y_predict)):
        driftDetector.add_element(y_predict[i] == y_test[i])
        if driftDetector.detected_change():
            drifts.append(i)
            driftDetector.reset()
    return drifts

def writeResults(f, driftDetectorName, drifts):
    l = len(drifts)
    f.write("%s: %d \\newline " % (driftDetectorName, l))
    for i in range(0, l):
        if (i != (l - 1)):
            f.write("%d, " % (drifts[i]))
        else:
            f.write("%d" % (drifts[i]))
    f.write("\n")

RECORDS_IN_CHUNK = 500 #2300
random_state=0
#classifier = RandomForestClassifier(random_state=0)
#classifier = GaussianNB()
#classifier = AdaBoostClassifier()
#classifier = DecisionTreeClassifier()
classifier = SVC()
#classifier = KNeighborsClassifier(n_neighbors=5)

file = "spam.arff"
#file = "electricity.arff"
#file = "spam.arff"
#file = "phishing.arff"
#file = "covtype.arff"
#file = "fin_digits08.arff"
#file ="fin_digits17.arff"

dataFrame = loadArffData(file)
accuracy = generateAccuracy(classifier, dataFrame, RECORDS_IN_CHUNK)
f = open("_ResultsConceptRecurringRT2.txt", "w")


# DDM, EDDM, ADWIN, HDDM_A, HDDM_W, KSWIN, PageHinkley
sum_elapse_time = 0
st = time.time()

driftDetector = DDM(min_num_instances = 30, warning_level = 2.0, out_control_level = 3.0)
drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "DDM", drifts)

#FDDM_OUTCONTROL = 0.9, FDDM_WARNING = 0.95, FDDM_MIN_NUM_INSTANCES = 30
driftDetector = EDDM()
drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
##writeResults(f, "EDDM", drifts)

#driftDetector = ADWIN(delta = 0.002)
#drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "ADWIN", drifts)

#driftDetector = HDDM_A(drift_confidence = 0.001, warning_confidence = 0.005, two_side_option = True)
#drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "HDDM_A", drifts)

#driftDetector = HDDM_W(drift_confidence = 0.001, warning_confidence = 0.005, lambda_option = 0.050, two_side_option = True)
#drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "HDDM_W", drifts)

#driftDetector = KSWIN(alpha = 0.005, window_size = 100, stat_size = 30, data = None)
#drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "KSWIN", drifts)

#driftDetector = PageHinkley(min_instances = 30, delta = 0.005, threshold = 50, alpha = 1 - 0.0001)
#drifts = generateDrifts(classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK)
#writeResults(f, "PageHinkley", drifts)
et = time.time()
elapsed_time = et - st


print('Drifts',len(drifts))
f.close()

for i in range(0, len(drifts)):
    drifts[i] = drifts[i] / RECORDS_IN_CHUNK

if len(drifts) > 0:
    plt.axvline(x = (drifts[0] - 1), color = 'b', linewidth=2)
    for i in range(1, len(drifts)):
        plt.axvline(x = (drifts[i] - 1), color = 'b', linewidth=2)

print('Accuracy', round(np.mean(accuracy),3))
print('Time=', elapsed_time)
#plt.plot(range(0, numberOfChunk), feat, 'b')
plt.ylim([0, 1])
plt.plot(range(0, len(accuracy)), accuracy, 'r')
plt.xlabel('Chunks', fontsize=12)
plt.ylabel('Accuracy (red)')
plt.tick_params(axis='both', which='major')
plt.legend()
plt.grid(True)
plt.savefig('Figure_1.pdf')
plt.show()

