import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(8888)
import graspy

from graspy.inference import LatentDistributionTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap, pairplot

#Graph 1-baseline n
n_verts = 200
P = np.array([[0.9, 0.11, 0.13, 0.2],
              [0, 0.7, 0.1, 0.1],
              [0, 0, 0.8, 0.1],
              [0, 0, 0, 0.85]])
P = symmetrize(P)
csize = [n_verts] * 4
A = sbm(csize, P)
ase = AdjacencySpectralEmbed(n_components=4)
X = ase.fit_transform(A)
ms = []
verts = []

#Graph 2 (n from 200 to 450)/p-value comparison
for n_verts1 in range(n_verts, n_verts+250, 50):
    csize1 = [n_verts1] * 4
    A1 = sbm(csize1, P)
    ase1 = AdjacencySpectralEmbed(n_components=4)
    X1 = ase1.fit_transform(A1)

    ldt = LatentDistributionTest(n_components=4)

    #run 15 tests and take the mean
    p = 0
    tests = 15
    for _ in range(tests):
        p += ldt.fit(A, A1)
    p /= tests
    print(p)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.hist(ldt.null_distribution_, 50)
    # ax.axvline(ldt.sample_T_statistic_, color='r')
    # ax.set_title("P-value = {} with {} vertices".format(p, n_verts1), fontsize=20)
    # plt.show()
    verts.append(n_verts1)
    ms.append(p)
print(verts)   
print(ms)
plt.xlabel("n_verts1")
plt.ylabel("p-value")
plt.title("Plot of P-Values with Different Cases of LDT (Baseline={})".format(n_verts))
plt.plot(verts, ms)
plt.show()
