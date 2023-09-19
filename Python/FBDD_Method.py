import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd                                              #ver. 1.5.3
import numpy as np                                               #ver. 1.23.5
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
#Lasso(normalize=True, tol=1e-5)
from skfeature.utility import construct_W

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

def StepwiseSelection(X, y, threshold_in = 0.01, threshold_out = 0.05, verbose = False):
    ''' Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    '''
    # forward step
    
    # Convert numpy array to DataFrame.
    Xdf = pd.DataFrame(X, columns = [str(i) for i in range(0, len(X[0]))])
    ydf = pd.DataFrame(y)
    
    initial_list = []
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(Xdf.columns) - set(included))
        new_pval = pd.Series(dtype = 'float64')
        for new_column in excluded:
            model = sm.OLS(ydf, sm.add_constant(pd.DataFrame(Xdf[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        
        # backward step
        model = sm.OLS(ydf, sm.add_constant(pd.DataFrame(Xdf[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    
    series = pd.concat([pvalues, new_pval])
    series = series.sort_values(ascending = True)
    l = series.index.tolist()
    l = [int(i) for i in l]
    
    return np.array(l)

def BordaRanking(X, y):
    # Methods initiation.
    featuresNumber = X.shape[1]
    instancesNumber = X.shape[0]
    
    C1, C2 = MakeClassSet(X, y, featuresNumber, instancesNumber)
    lasso = Lasso(alpha=0.001)
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 12, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # Scores.
    scores = []
    #scores.append(fisher_score.fisher_score(X, y, mode="index"))      #  1. Fisher score.
    #scores.append(KolmogorovSmirnov(C1, C2, featuresNumber))          #  2. Kolmogorov-Smirnov.
    #scores.append(TTest(C1, C2, featuresNumber))                      #  3. T-test.
    #scores.append(np.argsort(MIC(X, y, n_neighbors=12))[::-1])        #  5. Mutual information.
    #scores.append(CORR(X, C1, C2, featuresNumber))                    #  6. CORR.
    scores.append(np.argsort(lasso.fit(X, y).coef_))                  #  8. Lasso.
    #scores.append(lap_score.lap_score(X, y, mode="index", W=W))       # 10. Laplacian score.
    #scores.append(reliefF.reliefF(X, y, mode="index", W=W))           # 11. Relief 2.
    #scores.append(trace_ratio.trace_ratio(X, y, mode="index", W=W))   # 12. Trace ratio.
    #scores.append(StepwiseSelection(X, y))                            # 13. Step Wise Fit.

    #reliefFSScore = ReliefF()
    #fs = SelectKBest(f_classif, k=20)
    #lg = LogisticRegression(solver='lbfgs', max_iter=3000)
    #scores.append(reliefFSScore.fit(X, y).top_features_)              #  4. Relief 1.
    #scores.append(np.argsort(fs.fit(X, y).scores_)[::-1])             #  7. Anova. Kruskal-Wallis (KW).
    #scores.append((np.argsort(lg.fit(X, y).coef_)[::-1])[0])          #  9. Logistic regression.

    # Return Borda's ranking.
    #return mlpy.borda_count(scores)[0]
    return scores[0]

def loadArffData(fileName):
    classLabelEncoder = LabelEncoder()
    dataFrame = pd.DataFrame(loadarff(fileName)[0])
    i = len(dataFrame.columns)-1
    col = classLabelEncoder.fit_transform(dataFrame.iloc[:, i].values)
    dataFrame.isetitem(i, col)
    return dataFrame

def computeSTDinChunk(chunk, records_in_chunk):
    numberOfChunks = int(chunk.shape[0] / records_in_chunk)
    split = np.array_split(chunk, numberOfChunks)
    results = []
    featuresNumber = len(chunk.columns) - 1
    places = [0] * featuresNumber
    #f.write("******************************\n")
    for i in range(0, len(split)):
        X = split[i].drop(columns="class")
        y = split[i]["class"]

        #lasso = Lasso(alpha = 0.001)
        #results.append(np.argsort(lasso.fit(np.array(X), np.array(y)).coef_))
        #results.append(fisher_score.fisher_score(np.array(X), np.array(y), mode = "index"))

        results.append(BordaRanking(np.array(X), np.array(y)))

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

    feat = min(featureRanking, key = lambda t: t[1])
    return (feat[0], feat[3])

# Load data and split data into chunk.
#file = "spam.arff"
#file = "electricity.arff"
file = "spam.arff"
#file = "phishing.arff"
#file = "covtype.arff"
#file = "fin_digits08.arff"
#file ="fin_digits17.arff"
#file = "fin_wine.arff"

recordsInChunk = 500  # 1000 lub 2300
dataFrame = loadArffData(file)
numberOfChunk = int(np.ceil(dataFrame.shape[0] / recordsInChunk))
dataSplit = np.array_split(dataFrame, numberOfChunk)

(feat0, threshold) = computeSTDinChunk(dataSplit[0], recordsInChunk / 10)

# Train RandomForest classifier by chunk number 0.
X_train = dataSplit[0].drop(columns="class")
y_train = dataSplit[0]["class"]
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

drifts = []
results = []
feat = []
sum_elapse_time = 0
#accuracy without drift detetion
for i in range(1, numberOfChunk):
    X = dataSplit[i].drop(columns = "class")
    y = dataSplit[i]["class"]
    y_predict = classifier.predict(X)
    result = accuracy_score(y, y_predict)
    results.append(result)

    featuresNumber = len(X.columns)
    instancesNumber = len(X)
    
    st = time.time()
    score = BordaRanking(np.array(X), np.array(y))
    pos = np.where(score == feat0)
    et = time.time()
    elapsed_time = et - st
    sum_elapse_time = sum_elapse_time + elapsed_time

    if pos[0][0] > threshold:
        drifts.append(i)
        # Wyznaczenie rankingu raz jeszcze.
        if (i + 1) < numberOfChunk:
            (feat0, threshold) = computeSTDinChunk(dataSplit[i + 1], recordsInChunk / 10)

print("Accuracy: ", round(np.mean(results),3))
print("Number of drifts : ", len(drifts))
print("Our method time (sec): ", round(sum_elapse_time,3))
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
#plt.savefig("Figure_.pdf")
plt.show()

