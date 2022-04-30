import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 1.5)
y = x ** 5 + 2 * x ** 4 + x + 1
x_ = np.arange(-10, 10, 0.01)
y_ = x_ ** 5 + 2 * x_ ** 4 + x_ + 1


# ANS =  [1, 2, -3, 0, 1, 1]
# plt.plot(x, y, '.')
# plt.plot(x_, y_, 'r')
# plt.show()


def fitness(P, x, y):
    e = []
    for p in P:
        z = np.polyval(p, x)
        e.append(np.sum((z - y) ** 2))
    return e


def xover(p1, p2):
    x_point = np.random.randint(len(p1))
    c1 = np.concatenate([p1[:x_point], p2[x_point:]])
    c2 = np.concatenate([p2[:x_point], p1[x_point:]])
    return c1, c2


def mutate(p1):
    x_point = np.random.randint(len(p1))
    c1 = p1.copy()
    c1[x_point] += np.random.randn()
    return c1


degree = 5
n_pop = 100
n_sel = 50
P = np.random.randn(n_pop, degree)
n_gen = 1e4
i_gen = 0
while i_gen < n_gen:
    i_gen += 1
    # selection
    F = fitness(P, x, y)
    i = np.argsort(F)
    P = P[i]

    print('fitness = {}'.format(F[i[0]]))
    plt.clf(x, y, '.')  # problem
    plt.plot(x_, y_, 'r')  # solution
    z = np.polyval(P[0], x_)
    plt.plot(x_, z, 'r')  # GA
    plt.show(block=False)
    plt.pause(0.1)
    # Reproduction
    for j in range(n_sel, n_pop, 2):
        p1p2 = np.random.permutation(n_sel)[:2]
        P[j], P[j + 1] = xover(P[p1p2[0]], P[p1p2[1]])
    # Asexual
    for j in range(n_sel, n_pop):
        if np.random.rand() > 0.8:
            P[j] = mutate(P[np.random.randint(n_sel)])

# ANS = [1, 2, -3, 0, 1, 1]

