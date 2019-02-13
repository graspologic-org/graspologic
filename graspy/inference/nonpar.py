import numpy as np
import networkx as nx

from .base import BaseInference
from ..utils import import_graph, is_symmetric, symmetrize
from ..embed import select_dimension
from from sklearn.decomposition import TruncatedSVD as TSVD

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

    def _gaussian_covariance(X, Y, bandwidth = 0.5):
        diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
        return np.exp(-0.5 * np.sum(diffs**2, axis=2) / bandwidth**2)

    def _statistic(self, X, Y):
        N, _ = X.shape
        M, _ = Y.shape
        x_stat = np.sum(gaussian_covariance(X, X, 0.5) - np.eye(N))/(N*(N-1))
        y_stat = np.sum(gaussian_covariance(Y, Y, 0.5) - np.eye(M))/(M*(M-1))
        xy_stat = np.sum(gaussian_covariance(X, Y, 0.5))/(N*M)
        return x_stat - 2*xy_stat + x_stat

    def _ase(self, A):
        tsvd = TSVD()
        vecs, vals = tsvd.fit(A).components_, tsvd.singular_values_
        vecs_2 = np.array([vecs[0, :], vecs[1, :]])
        if vecs_2[0,0] < 0:
            vecs_2 *= -1
        X_hat = vecs_2.T @ np.diag(vals[:2]**(1/2))
        return X_hat

    def _bootstrap(X, Y, M, alpha = 0.05):
        N, _ = X.shape
        M, _ = Y.shape

        statistics = np.zeros(M)
        for i in range(M):
            bs_X = X[np.random.choice(np.arange(0,N), size = int(N/2), replace = False)]
            bs_Y = Y[np.random.choice(np.arange(0,M), size = int(M/2), replace = False)]
            statistics[i] = statistic(bs_X, bs_Y)

        sorted_ = np.sort(statistics)
        rej_ind = int(np.ceil(((1 - alpha)*M)))
        return sorted_[rej_ind]

    # TODO calculate and return p-value
    def estimated_power(n, eps, M, alpha, iters):
        sizes, probsA, probsB, A1, A2 = gen_data(n, eps)

        X1_hat = ASE(A1)
        X2_hat = ASE(A2)
        critical_value = bootstrap(X1_hat, X2_hat, M, alpha)

        rejections = 0
        for i in range(iters):
            G3 = nx.stochastic_block_model(sizes, probsA)
            A = nx.to_numpy_array(G3)
            G4 = nx.stochastic_block_model(sizes, probsB)
            B = nx.to_numpy_array(G4)
            X_hat = ASE(A)
            Y_hat = ASE(B)

            U = statistic(X_hat, Y_hat)
            if U > critical_value:
                rejections += 1
        return rejections/iters
