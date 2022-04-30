from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.font_manager as fm
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt

data = loadmat('HR1.mat')
data['dep'][data['dep'] == ' -'] = '-'
for ind, i in enumerate(data['dep']):
    data['dep'][ind][0] = i[0][0]
unique_dep, counts = np.unique(data['dep'], return_counts=True)
for ind, i in enumerate(data['sex']):
    data['sex'][ind][0] = i[0][0]
unique_sex, counts = np.unique(data['sex'], return_counts=True)

fpath = './THSarabunNew.ttf'
prop = fm.FontProperties(fname=fpath)


def fitness(P, dataset, n_group=2):
    e = []
    for p in P:
        dep_ = dict.fromkeys(unique_dep, 0)
        sex_ = dict.fromkeys(unique_sex, 0)
        for i in range(n_group):
            group = np.where(p == i)
            filter = dataset[group]
            unique, counts = np.unique(filter[:, 0], return_counts=True)
            for key, var in zip(unique, counts):
                dep_[key] = abs(dep_[key] - var)
            unique, counts = np.unique(filter[:, 1], return_counts=True)
            for key, var in zip(unique, counts):
                sex_[key] = abs(sex_[key] - var)

        e_dep = sum(dep_.values()) / len(dep_.values())
        e_sex = sum(sex_.values()) / len(sex_.values())

        e.append((e_dep) + (e_sex))
    return e


def mutate(p1):
    c1 = p1.copy()
    x_all, y_all = np.where(p1 == 0)[0], np.where(p1 == 1)[0]
    x, y = np.random.choice(x_all), np.random.choice(y_all)
    c1[x], c1[y] = c1[y], c1[x]
    return c1


def xover(p1, p2):
    x_point = np.random.randint(len(p1))
    c1 = np.concatenate([p1[:x_point], p2[x_point:]])
    c2 = np.concatenate([p2[:x_point], p1[x_point:]])
    return c1, c2


dataset = np.concatenate((data['dep'], data['sex']), axis=1)
n_pop, n_gene = 100, data['dep'].size
n_group = 2
P = np.random.randint(0, n_group, (n_pop, n_gene))

i_gen, n_gen = 0, 200  # var loop control

n_sel = n_pop // 2  # var reproduction

# var visualization
error_arr = []
last_fitness = 0

while i_gen < n_gen:
    i_gen += 1
    # selection
    F = fitness(P, dataset)
    i = np.argsort(F)
    P = P[i]
    error_fitness = F[i[0]]

    # visualization
    error_arr.append(error_fitness)

    dep_ = []
    sex_ = []
    for i in range(n_group):
        dep_v = dict.fromkeys(unique_dep, 0)
        sex_v = dict.fromkeys(unique_sex, 0)
        group = np.where(P[0] == i)
        filter = dataset[group]
        unique, counts = np.unique(filter[:, 0], return_counts=True)
        for key, var in zip(unique, counts):
            dep_v[key] = var
        dep_.append(dep_v)
        unique, counts = np.unique(filter[:, 1], return_counts=True)
        for key, var in zip(unique, counts):
            sex_v[key] = var
        sex_.append(sex_v)

    print(f'Epoch : {i_gen}/{n_gen}')
    print('fitness = {}'.format(error_fitness))
    if last_fitness != error_fitness:
        # print(f'Epoch : {i_gen}/{n_gen}')
        # print('fitness = {}'.format(error_fitness))

        df = pd.DataFrame(sex_, index=list(range(1, n_group + 1)))
        df = df.T
        df.plot(kind="bar")
        plt.savefig('img1.png')

        df = pd.DataFrame(dep_, index=list(range(1, n_group + 1)))
        df = df.T
        df.plot(kind="barh")
        plt.savefig('img2.png')
        plt.close('all')

        image = cv2.imread('./img1.png')
        image2 = cv2.imread('./img2.png')
        cv2.imshow('img1', image)
        cv2.imshow('img2', image2)
        cv2.waitKey(1)
        last_fitness = error_fitness

    # weighted_random_choice
    # randomNumberList = random.choices(
    #     P, weights=list(range(n_pop, 0, -1)), k=n_pop)
    # P = np.array(randomNumberList)

    p = np.arange(n_pop, 0, -1)
    p = p / p.sum()
    randomNumberList = np.random.choice(
        np.arange(n_pop), n_pop, p=p)
    P = P[randomNumberList]

    # Reproduction
    for j in range(n_sel, n_pop, 2):
        p1p2 = np.random.permutation(n_sel)[:2]
        P[j], P[j + 1] = xover(P[p1p2[0]], P[p1p2[1]])
        # P[j], P[j + 1] = xover(P[j], P[j + 1])

    # Asexual
    for j in range(n_sel, n_pop):
        if np.random.rand() > 0.5:
            P[j] = mutate(P[np.random.randint(n_sel)])

plt.plot(error_arr)
plt.show()
# plt.savefig('error.png')
