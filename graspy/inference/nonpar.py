# Bijan Varjavand
# bpedigo [at] jhu.edu
# 10.18.2018

import numpy as np
import networkx as nx

from .base import BaseInference
from ..utils import import_graph, is_symmetric, symmetrize
from ..embed import select_dimension, AdjacencySpectralEmbed
from sklearn.decomposition import TruncatedSVD as TSVD

class NonparametricTest(BaseInference):
    """
    Two sample hypothesis test for the nonparamatric problem of determining
    whether two random dot product graphs have the same latent positions.

    Parameters
    ----------
    embedding : string, { 'ase' (default), 'lse', 'omnibus'}
        String describing the embedding method to use.
        Must be one of:
        'ase'
            Embed each graph separately using adjacency spectral embedding
            and use Procrustes to align the embeddings.
        'lse'
            Embed each graph separately using laplacian spectral embedding
            and use Procrustes to align the embeddings.
        'omnibus'
            Embed all graphs simultaneously using omnibus embedding.

    n_components : None (default), or Int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    n_bootstraps : 200 (default), or Int
        Number of bootstrap iterations.
    """
    # TODO
    #     - import graphs properly in the class-based structure
    #     - "fix" LCC, then use graspy ASE
    def __init__(self,
                 embedding='ase',
                 n_components=None,
                 n_bootstraps=200,):
        if type(n_bootstraps) is not int:
            raise TypeError()

        if n_bootstraps < 1:
            msg = '{} is invalid number of bootstraps, must be greater than 1'
            raise ValueError(msg.format(n_bootstraps))

        super().__init__(
            embedding=embedding,
            n_components=n_components,
        )
        # self.embedding = embedding
        # self.n_components = n_components
        self.n_bootstraps = n_bootstraps

    def _gaussian_covariance(self, X, Y, bandwidth = 0.5):
        diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
        return np.exp(-0.5 * np.sum(diffs**2, axis=2) / bandwidth**2)

    def _statistic(self, X, Y):
        N, _ = X.shape
        M, _ = Y.shape
        x_stat = np.sum(self._gaussian_covariance(X, X, 0.5) - np.eye(N))/(N*(N-1))
        y_stat = np.sum(self._gaussian_covariance(Y, Y, 0.5) - np.eye(M))/(M*(M-1))
        xy_stat = np.sum(self._gaussian_covariance(X, Y, 0.5))/(N*M)
        return x_stat - 2*xy_stat + x_stat

    def _ase(self, A, max_d):
        ase = AdjacencySpectralEmbed(n_components = max_d, algorithm = 'randomized')
        X_hat = ase.fit_transform(A)
        return X_hat
    
    def _median_heuristic(self, X1, X2):
        X1_medians = np.median(X1, axis=0)
        X2_medians = np.median(X2, axis=0)
        val = np.multiply(X1_medians, X2_medians)
        t = (val>0)*2-1
        X1 = np.multiply(t.reshape(-1,1).T,X1)
        return X1, X2

    def _bootstrap(self, X, Y, M = 200):
        N, _ = X.shape
        M2, _ = Y.shape
        Z = np.concatenate((X,Y))
        statistics = np.zeros(M)
        for i in range(M):
            bs_Z = Z[np.random.choice(np.arange(0,N+M2), size = int(N+M2), replace = False)]
            bs_X2 = bs_Z[:N,:]
            bs_Y2 = bs_Z[N:,:]
            statistics[i] = self._statistic(bs_X2, bs_Y2)
        return statistics

    def fit(self, A1, A2):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.

        Returns
        -------
        p : float
            The p value corresponding to the specified hypothesis test
        """
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        A1_d = select_dimension(A1)[0][-1]
        A2_d = select_dimension(A2)[0][-1]
        max_d = max(A1_d, A2_d)
        X1_hat = self._ase(A1, max_d)
        X2_hat = self._ase(A2, max_d)
        X1_hat, X2_hat = self._median_heuristic(X1_hat, X2_hat)
        U = self._statistic(X1_hat, X2_hat)
        null_distribution = self._bootstrap(X1_hat, X2_hat, self.n_bootstraps)

        self.null_distribution_ = null_distribution
        self.sample_T_statistic_ = U
        p_value = (
            len(null_distribution[null_distribution >= U]) + 0.5
        ) / self.n_bootstraps
        self.p_value_ = p_value
        return p_value
