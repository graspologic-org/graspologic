#%%
from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed

# from .base import BaseSpectralEmbed
# from ..utils import import_graph, to_laplacian
import numpy as np


class CASC(BaseSpectralEmbed):
    # TODO: everything
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_fitted_ = False

    def fit(self, graph, covariates, y=None):
        self.is_fitted_ = True
        X = import_graph(graph)
        L = to_laplacian(X)

    def transform(self, graph, y=None):
        A = import_graph(graph)
        pass


#%%
from graspologic.embed.ase import AdjacencySpectralEmbed

# A good initial choice of a is the value which makes the leading eigenvalues of LL and aXX^T equal, namely
# a_0 = \lambda_1 (LL) / \lambda_1 (XX^T)
Lsquared = L @ L
Lleading = sorted(np.linalg.eigvals(Lsquared), reverse=True)[0]
Xleading = sorted(np.linalg.eigvals(X @ X.T), reverse=True)[0]
a = np.float(Lleading / Xleading)
L_ = (L @ L) + (a * (X @ X.T))
# heatmap(L_)
# heatmap(Lsquared)
# heatmap(X @ X.T)

ase = AdjacencySpectralEmbed(n_components=2)
ase._reduce_dim(L)
X = ase.latent_left_


def gen_covariates(m1, m2, labels):
    # TODO: make sure labels is 1d array-like
    n = len(labels)
    m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
    m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, 3))
    m2_arr[np.arange(n), labels] = m1_arr
    return m2_arr


#%%
#%%
# gonna do this in code as I read

import numpy as np
import graspologic as gs
from graspologic.simulations import sbm
from graspologic.plot import heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def gen_covariates(m1, m2, labels):
    # TODO: make sure labels is 1d array-like
    n = len(labels)
    m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
    m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, 3))
    m2_arr[np.arange(n), labels] = m1_arr
    return m2_arr


#%%

n = 200
n_communities = 3
p, q = 0.9, 0.3
B = np.array([[p, q, q], [q, p, q], [q, q, p]])

B2 = np.array([[q, p, p], [p, q, p], [p, p, q]])

A, labels = sbm([n, n, n], B, return_labels=True)
N = A.shape[0]
L = gs.utils.to_laplace(A, form="R-DAD")
X = gen_covariates(0.9, 0.1, labels)


heatmap(L)
# heatmap(L)
#%%
sns.heatmap(X)
#%%
from graspologic.embed.base import BaseSpectralEmbed
from graspologic.embed.svd import selectSVD


bse = BaseSpectralEmbed(n_components=3)
U, D, V = selectSVD(L, n_components=3)
U.shape
D.shape
V.shape
# %%
# %%

from graspologic.embed.ase import AdjacencySpectralEmbed

# A good initial choice of a is the value which makes the leading eigenvalues of LL and aXX^T equal, namely
# a_0 = \lambda_1 (LL) / \lambda_1 (XX^T)
Lsquared = L @ L
Lleading = sorted(np.linalg.eigvals(Lsquared), reverse=True)[0]
Xleading = sorted(np.linalg.eigvals(X @ X.T), reverse=True)[0]
a = np.float(Lleading / Xleading)
L_ = (L @ L) + (a * (X @ X.T))
# heatmap(L_)
# heatmap(Lsquared)
# heatmap(X @ X.T)

ase = AdjacencySpectralEmbed(n_components=2)
ase._reduce_dim(L)
X = ase.latent_left_

scatter = plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.gcf().set_size_inches(5, 5)
plt.gca().legend(*scatter.legend_elements())
plt.title(r"Spectral embedding of $LL + aXX^T$")
plt.savefig("/Users/alex/Dropbox/School/NDD/graspy-personal/figs/casc_working.png")

#%%
# heatmap(L_)
heatmap(X @ X.T)
