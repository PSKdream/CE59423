from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib.font_manager as fm

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt

data = loadmat('HR1.mat')
data['dep'][data['dep'] == ' -'] = '-'
# print(data['dep'])
for ind, i in enumerate(data['dep']):
    data['dep'][ind][0] = i[0][0]
unique_dep, counts = np.unique(data['dep'], return_counts=True)
dep = np.column_stack((unique_dep, counts))
for ind, i in enumerate(data['sex']):
    data['sex'][ind][0] = i[0][0]
data['sex'][data['sex'] == 'ช'] = 'Male'
data['sex'][data['sex'] == 'ญ'] = 'Female'
unique_sex, counts = np.unique(data['sex'], return_counts=True)
sex = np.column_stack((unique_sex, counts))

dep_, sex_ = dep.copy(), sex.copy()
# print(unique_sex)

def fitness(P, dataset, n_group=2):
    e = []
    for p in P:
        dep_[:, 1] = 0
        sex_[:, 1] = 0
        for i in range(n_group):
            group = np.where(p == i)
            filter = dataset[group]
            unique, counts = np.unique(filter[:, 0], return_counts=True)
            # print(unique, counts)
            for key, var in zip(unique, counts):
                ind = np.where(dep_[:, 0] == key)
                dep_[ind, 1] = abs(dep_[ind][0, 1] - var)
            unique, counts = np.unique(filter[:, 1], return_counts=True)
            for key, var in zip(unique, counts):
                ind = np.where(sex_[:, 0] == key)
                sex_[ind, 1] = abs(sex_[ind][0, 1] - var)

        e_dep = np.sum(dep_[:, 1]) / dep_[:, 1].size
        e_sex = np.sum(sex_[:, 1]) / sex_[:, 1].size

        e.append((e_dep) + (e_sex))
    return e


def mutate(p1):
    for _ in range(int(len(p1) / 0.05)):
        x, y = np.where(p1 == 0), np.where(p1 == 1)
        x, y = np.random.choice(x[0]), np.random.choice(y[0])
        c1 = p1.copy()
        temp = c1[x]
        c1[x] = c1[y]
        c1[y] = temp
    return c1


dataset = np.concatenate((data['dep'], data['sex']), axis=1)
n_pop, n_gene = 100, data['dep'].size
# print(n_pop, n_gene)
P = np.random.randint(0, 2, (n_pop, n_gene))
n_gen = 100
i_gen = 0
n_sel = n_pop // 2
n_group = 2
error_arr = []
# print(n_sel, n_pop)
while i_gen < n_gen:
    print(f'Epoch : {i_gen}/{n_gen}')
    i_gen += 1
    # selection
    F = fitness(P, dataset)
    i = np.argsort(F)
    P = P[i]
    error_arr.append(F[i[0]])
    print('fitness = {}'.format(F[i[0]]))

    # visualization
    dep_v, sex_v = [], []
    for i in range(n_group):
        group = np.where(P[0] == i)
        filter = dataset[group]
        unique, counts = np.unique(filter[:, 0], return_counts=True)
        dep_v.append(np.column_stack([unique, counts]))
        unique, counts = np.unique(filter[:, 1], return_counts=True)
        sex_v.append(np.column_stack([unique, counts]))

    df = pd.DataFrame([])
    for ind, i in enumerate(sex_v):
        # print(i)
        df2 = pd.DataFrame([i[:, 1]], columns=unique_sex, index=[ind])
        df = pd.concat([df, df2])
    # print(df)

    df.plot(kind="bar")
    plt.savefig('img1.png')

    fpath = './THSarabunNew.ttf'
    prop = fm.FontProperties(fname=fpath)
    g1 = dep_v[0][:, 1]
    g2 = dep_v[1][:, 1]
    index = unique_dep
    df = pd.DataFrame({'group1': g1,
                       'group2': g2}, index=index)
    df.plot(kind="barh")
    plt.savefig('img2.png')

    image = cv2.imread('./img1.png')
    image2 = cv2.imread('./img2.png')
    image_error = cv2.imread('./error.png')
    cv2.imshow('img1', image)
    cv2.imshow('img2', image2)
    cv2.waitKey(1)
    # Asexual
    # print(1111)
    for j in range(n_sel, n_pop):
        # if np.random.rand() > 0.7:
        P[j] = mutate(P[np.random.randint(n_sel)])

# plt.plot(error_arr)
# plt.show()
# plt.savefig('error.png')
