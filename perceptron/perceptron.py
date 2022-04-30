import numpy as np

X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
Y = [1, -1, -1, -1]

X_ = np.hstack((X, np.ones((len(X), 1))))
for i in range(len(X)):
    X_[i] = X_[i] * Y[i]
W = X_[np.random.randint(len(X))]

c = 0
while c < len(X):
    c = 0
    for x in X_:
        if np.dot(W, x) < 0:
            W = W + x
            print(W)
        else:
            c += 1

print('Solution', W)
X_ = np.hstack((X, np.ones((len(X), 1))))
Z = X_ @ W[:, None]
Z[Z >= 0] = 1
Z[Z < 0] = -1
print('Y', Y)
print('Z', Z.flatten())
