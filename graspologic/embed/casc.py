#%%
from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed
from graspologic.embed.lse import LaplacianSpectralEmbed
from graspologic.simulations import sbm
from graspologic.plot import heatmap

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns


class CASC(BaseSpectralEmbed):
    # TODO: everything
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.assortive_ = False
        self.is_fitted_ = False

    def fit(self, graph, covariates, y=None):
        # setup
        A = import_graph(graph)
        X = covariates.copy()

        # workhorse code
        L = to_laplacian(A, form="R-DAD")
        LL = L if self.assortive_ else L @ L
        XX = X @ X.T
        a = self._get_tuning_parameter(LL, XX)
        L_ = LL + a * (XX)
        self._reduce_dim(L_)

        self.is_fitted_ = True
        return self

    def transform(self, graph, y=None):
        A = import_graph(graph)
        pass

    def _get_tuning_parameter(self, LL, XX):
        """
        Find an a which causes the leading eigenvectors of L@L and a*X@X.T to be the same.
        """
        L_leading = np.linalg.eigvalsh(LL)[-1]
        X_leading = np.linalg.eigvalsh(XX)[-1]
        return np.float(L_leading / X_leading)


def gen_covariates(m1, m2, labels, ndim=3, static=False):
    # TODO: make sure labels is 1d array-like
    n = len(labels)

    if static:
        m1_arr = np.full(n, m1)
        m2_arr = np.full((n, ndim), m2)
        m2_arr[np.arange(n), labels] = m1_arr
    elif not static:
        m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
        m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, ndim))
        m2_arr[np.arange(n), labels] = m1_arr

    return m2_arr
