import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def knn(x, X, Y, k=1):
    p = 3
    distance = np.power(np.sum(abs(X - x) ** p, axis=1), (1 / p))
    arg = distance.argsort()
    y_set = Y[arg[:k]]
    unique, pos = np.unique(y_set, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    return unique[maxpos]


def FuzzyCmeans(X, k, m=2, th=1e-6):
    # init
    ik = np.random.permutation(len(X))[:k]
    C = X[ik].copy()
    C_ = 0
    while np.mean(np.abs(C_ - C)) > th:
        mu = []
        for x in X:
            d = np.sum((C - x) ** 2, axis=1)
            mu.append((1 / d) ** (1 / (m - 1)))
        mu = np.array(mu)
        mu = mu.T / np.sum(mu, axis=1)
        mu[np.isnan(mu)] = 1
        # mu = mu.T
        # update C
        C_ = C.copy()
        for k in range(len(C)):
            w = mu[k] ** m
            C[k] = np.sum(X.T * w, axis=1) / np.sum(w)

    return C, mu


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
X = df.iloc[:, :4].values
Y = df.iloc[:, -1].values
classes = np.unique(Y)

itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]
Xtrain, Ytrain = X[itrain], Y[itrain]
Xtest, Ytest = X[itest], Y[itest]

C, mu = FuzzyCmeans(Xtrain, 3)
r = np.argmax(mu, axis=0)
Ctrue = []
for i in [0, 25, 50]:
    Ctrue.append(np.mean(X[i:i + 25], axis=0))
Ctrue = np.array(Ctrue)
print(Ctrue)
itrue = []
classes_ = []
for c in C:
    d = np.sum((Ctrue - c) ** 2, axis=1)
    i = np.argmin(d)
    itrue.append(i)
    classes_.append(classes[i])
print(itrue)
classes_ = np.array(classes_)
print(classes_)
Z = classes_[r]
print(Z)
print(np.sum(Z == Ytrain) / len(Z))
