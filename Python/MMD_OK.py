############################
# Python ver. 3.8
# WORKS!!
############################
import warnings
import pandas as pd  # ver. 1.5.3
import numpy as np  # ver. 1.23.5
import matplotlib.pyplot as plt
from numpy import mean
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.io.arff import loadarff
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from frouros.unsupervised.distance_based import MMD  # ver. 0.1.0

warnings.filterwarnings("ignore")

np.random.seed(0)


def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    data_frame = pd.DataFrame(loadarff(fileName)[0])
    i = len(data_frame.columns) - 1
    col = classLabelEncoder.fit_transform(data_frame.iloc[:, i].values)
    data_frame.isetitem(i, col)
    return data_frame


recordsInChunk = 500  # 2300
# Load data *.arff and split data into chunk.
# file = "abrupt_HP_10.arff"
# file = "abrupt_HP_20.arff"
# file = "abrupt_HP_30.arff"
# file = "abrupt_HP_40.arff"
# file = "abrupt_HP_50.arff"

# file = "abrupt_RT_10.arff"
# file = "abrupt_RT_20.arff"
# file = "abrupt_RT_30.arff"
# file = "abrupt_RT_40.arff"
# file = "abrupt_RT_50.arff"

# file = "abrupt_RBF_10.arff"
# file = "abrupt_RBF_20.arff"
# file = "abrupt_RBF_30.arff"
# file = "abrupt_RBF_40.arff"
# file = "abrupt_RBF_50.arff"
###########################################################
# file = "gradual_HP_10.arff"
# file = "gradual_HP_20.arff"
# file = "gradual_HP_30.arff"
# file = "gradual_HP_40.arff"
# file = "gradual_HP_50.arff"

# file = "gradual_RT_10.arff"
# file = "gradual_RT_20.arff"
# file = "gradual_RT_30.arff"
# file = "gradual_RT_40.arff"
# file = "gradual_RT_50.arff"

# file = "gradual_RBF_10.arff"
# file = "gradual_RBF_20.arff"
# file = "gradual_RBF_30.arff"
# file = "gradual_RBF_40.arff"
# file = "gradual_RBF_50.arff"
#########################################################
# file = "recurring_HP_10.arff"
# file = "recurring_HP_20.arff"
# file = "recurring_HP_30.arff"
# file = "recurring_HP_40.arff"
# file = "recurring_HP_50.arff"

# file = "recurring_RT_10.arff"
# file = "recurring_RT_20.arff"
# file = "recurring_RT_30.arff"
# file = "recurring_RT_40.arff"
# file = "recurring_RT_50.arff"

# file = "recurring_RBF_10.arff"
# file = "recurring_RBF_20.arff"
# file = "recurring_RBF_30.arff"
# file = "recurring_RBF_40.arff"
# file = "recurring_RBF_50.arff"
##################################################
# file= "electricity.arff"
# file = "wine_quality.arff"
# file = "spam.arff"
# file = "covtype.arff"
file = "phishing.arff"
# file = "fin_digits17.arff"
# file = "fin_digits08.arff"
dataFrame = loadArffData(file)

numberOfChunks = int(dataFrame.shape[0] / recordsInChunk)
dataSplit = np.array_split(dataFrame, numberOfChunks)

# Train Classifier by chunk number 0.
X_ref = dataSplit[0].drop(columns="class")
y_ref = dataSplit[0]["class"]

# Classifier selection & classifier learn
random_state = 0
classifier = RandomForestClassifier()
# classifier = GaussianNB()
# classifier = AdaBoostClassifier()
# classifier = DecisionTreeClassifier()
# classifier = SVC()
# classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_ref, y_ref)

results = []
results_After = []
Total = []
# accuracy without drift detection
for i in range(1, numberOfChunks):
    X = dataSplit[i].drop(columns="class")
    y = dataSplit[i]["class"]
    y_predict = classifier.predict(X)
    Acc = accuracy_score(y, y_predict)
    results.append(Acc)
avg_ACC = np.mean(results)

alpha = 0.01  # significance level for the hypothesis test
detector = MMD(num_permutations=100, kernel=RBF(length_scale=1.0), random_state=0)
detector.fit(X_ref)
k = 0
for i in range(1, numberOfChunks):
    X = dataSplit[i].drop(columns="class")
    y = dataSplit[i]["class"]

    detector.transform(X)
    mmd, p_value = detector.distance
    y_predict = classifier.predict(X)
    if p_value < alpha:
        k = k + 1
        print("Drift detected @ chunk:", i)
        classifier.fit(X, y)
        detector.fit(X)
        X_ref = X
    Acc = accuracy_score(y, y_predict)
    results_After.append(Acc)
avg_ACC_Drift = sum(results_After) / len(results_After)
avg_ACC_Drift = round(avg_ACC_Drift, 3)

print("Accuracy without drift detection =", round(avg_ACC, 3))
print("Accuracy with drift detection:", avg_ACC_Drift)
print("Drifts =", k)

plt.subplot(2, 1, 1)
plt.plot(range(0, len(results)), results, "r", linewidth=1)
plt.xlabel("Chunks", fontsize=12)
plt.ylabel("Accuracy before", fontsize=12)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.axis([0, len(results), 0, 1.05])
plt.grid(True)
# plt.show()

plt.subplot(2, 1, 2)
plt.plot(range(0, len(results_After)), results_After, "r", linewidth=1)
plt.xlabel("Chunks", fontsize=12)
plt.ylabel("Accuracy after", fontsize=12)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.axis([0, len(results_After), 0, 1.05])
plt.grid(True)
plt.show()
