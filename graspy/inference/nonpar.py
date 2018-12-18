import numpy as np
from scipy.spatial.distance import pdist

from .base import BaseInference
from ..utils import import_graph, is_symmetric, symmetrize
from ..embed import select_dimension

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

    def _bootstrap(self, x, y):
        z = np.concatenate((x, y))
        boots = []
        for _ in range(self.n_bootstraps):
            np.random.shuffle(z)
            u = self._u_from_z(z, len(x))
            boots.append(u)
        return boots

    def _u_from_z(self, z, n):
        k = self._dist_matrix(z)
        gk = np.exp(-k / np.median(k))
        u = np.mean(gk[:n, :n]) + np.mean(gk[n:, n:]) - 2 * np.mean(gk[:n, n:])
        return u

    def _dist_matrix(self, z):
        dists = pdist(z, 'euclidean')
        zlen = len(z)
        ind = np.triu_indices(zlen, k=1)
        k = np.zeros((zlen, zlen))
        k[ind] = dists
        return symmetrize(k)

    def _median_heuristic(self, X_hats):
        X1_hat = X_hats[0]
        X2_hat = X_hats[1]
        X1_med = np.median(X1_hat, axis=0)
        X2_med = np.median(X2_hat, axis=0)
        # if sign of median is off, multiply by -1
        opp = np.multiply(X1_med, X2_med) >= 0
        opp2 = [int(o) * 2 - 1 for o in opp]  # 1 if mean same sign, -1 if not
        X2_hat = np.multiply(X2_hat, opp2)
        return (X1_hat, X2_hat)

    def fit(self, A1, A2):
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        if not is_symmetric(A1) or not is_symmetric(A2):
            raise NotImplementedError()

        if self.n_components is None:
            num_dims1 = select_dimension(A1)[0][-1]
            num_dims2 = select_dimension(A2)[0][-1]
            self.n_components = max(num_dims1, num_dims2)

        X_hats = self._embed(A1, A2)
        X_hats = self._median_heuristic(X_hats)
        U_sample = self._u_from_z(np.concatenate(X_hats), len(X_hats[0]))
        U_bootstrap = self._bootstrap(X_hats[0], X_hats[1])

        self.U_bootstrap = U_bootstrap
        self.U_sample = U_sample
        return U_sample
