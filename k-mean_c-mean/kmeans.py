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


def kmeans(X, k):
    # init
    ik = np.random.permutation(len(X))[:k]
    C = X[ik].copy()
    C_ = 0
    while np.mean(np.abs(C_ - C)) > 0:
        r = []
        for x in X:
            d = np.sum((C - x) ** 2, axis=1)
            r.append(np.argmin(d))
        r = np.array(r)
        # update C
        C_ = C.copy()
        for i in range(len(C)):
            C[i] = np.mean(X[r == i], axis=0)

    return C, r


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
X = df.iloc[:, :4].values
Y = df.iloc[:, -1].values
classes = np.unique(Y)

itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]
Xtrain, Ytrain = X[itrain], Y[itrain]
Xtest, Ytest = X[itest], Y[itest]

C, r = kmeans(Xtrain, 3)
Ctrue = []
for i in [0, 25, 50]:
    Ctrue.append(np.mean(X[i:i+25], axis=0))
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