import numpy as np
from scipy.spatial.distance import pdist

from .base import BaseInference
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed


class NonparametricTest(BaseInference):
    """
    Two sample hypothesis test for the nonparametric problem of determining
    whether two random dot product graphs have the same underlying
    distributions.

    Parameters
    ----------
    embedding : { 'ase' (default), 'lse, 'omnibus'}
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

    monte_iter : 1000 (default), or Int
        Number of monte carlo iterations.
    """

    def __init__(self, embedding='ase', n_components=None, n_bootstraps=200, monte_iter=1000):
        if type(n_bootstraps) is not int:
            raise TypeError()
        if n_bootstraps < 1:
            raise ValueError('Must have a positive number of bootstraps, not {}'.format(n_bootstraps))
        super().__init__(embedding=embedding, n_components=n_components,)
        self.embedding = embedding
        self.n_bootstraps = n_bootstraps

    def _embed(self, A1, A2):
        if self.embedding not in ['ase', 'lse', 'omnibus']:
            raise ValueError('Invalid embedding method "{}"'.format(self.embedding))
        if self.embedding == 'ase':
            X1_hat = AdjacencySpectralEmbed(k=self.n_components).fit_transform(A1)
            X2_hat = AdjacencySpectralEmbed(k=self.n_components).fit_transform(A2)
        elif self.embedding == 'lse':
            X1_hat = LaplacianSpectralEmbed(k=self.n_components).fit_transform(A1)
            X2_hat = LaplacianSpectralEmbed(k=self.n_components).fit_transform(A2)
        elif self.embedding == 'omnibus':
            X_hat_compound = OmnibusEmbed(k=self.n_components).fit_transform((A1, A2))
            X1_hat = X_hat_compound[:A1.shape[0],:]
            X2_hat = X_hat_compound[A2.shape[0]:,:]
        return (X1_hat, X2_hat)

    def _gen_x1hat_x2hat(self,X1,X2):
        #generate x matrix nxd and y matrix nxd
        #generate A = bern(XXt) B = bern(YYt)
        P1 = p_from_latent(X1)
        A1 = rdpg_from_p(P1)
        P2 = p_from_latent(X2)
        A2 = rdpg_from_p(P2)
        X1_hat, X2_hat = self._embed(A1, A2)
        return (X1_hat, X2_hat)

    def _gen_kernel_embedding(self,Z):
        dists = pdist(Z.T, 'euclidean')
        all_d = np.concatenate((dists,dists,np.zeros(len(dists))))
        dists = np.exp(-dists/np.median(all_d))

        zlen = z.shape[1]
        ind = np.triu_indices(zlen,k=1)
        k = np.zeros((zlen,zlen))
        k[ind] = dists
        return k

    def _bootstrap(self, X1, X2):
        t_bootstrap = np.zeros(self.n_bootstraps)
        for i in range(self.n_bootstraps):
            Z = np.concatenate((X1,X2),dim=1)
            Z = Z[np.random.shuffle(np.transpose(Z))] #shuffle
            k = self._gen_kernel_embedding(Z)
            n = X1.shape[1]
            m = X2.shape[1]
            dist = np.mean(k[:n,:n]) + np.mean(k[n:,n:]) - 2*np.mean(k[n:,:n])
            t_bootstrap[i] = dist
        return t_bootstrap

    def fit(self, X1, X2):
        #t_monte = np.zeros(self.monte_iter) TODO p-value
        #for i in range(self.monte_iter):
        t_bootstrap = self._bootstrap(X1,X2)
        X1_hat, X2_hat = self._gen_x1hat_x2hat(X,Y)
        dist_hat = pdist(np.concatenate((X1_hat,X2_hat),axis=1).T, 'euclidean')

        self.bootstrap = t_bootstrap
        self.sample = dist_hat
        #self.p = p
        return #p
