# Ben Pedigo
# bpedigo [at] jhu.edu
# 10.18.2018

import numpy as np

from .base import BaseInference
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from ..simulations import rdpg_from_p, p_from_latent
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from ..utils import import_graph, is_symmetric

class SemiparametricTest(BaseInference):
    """
    Two sample hypothesis test for the semiparamatric problem of determining
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

    test_case : string, {'rotation (default), 'scalar-rotation', 'diagonal-rotation'}
        describes the exact form of the hypothesis to test when using 'ase' or 'lse' 
        as an embedding method. Ignored if using 'omnibus'
    """

    def __init__(self, embedding='ase', n_components=2, n_bootstraps=1000, test_case='rotation',):

        if type(n_bootstraps) is not int:
            raise TypeError()
        if type(test_case) is not str:
            raise TypeError()

        if n_bootstraps < 1:
            raise ValueError('{} is invalid number of bootstraps, must be greater than 1'.format(n_bootstraps))
        if test_case not in ['rotation', 'scalar-rotation', 'diagonal-rotation']:
            raise ValueError('test_case must be one of \'rotation\', \'scaling-rotation\',\'diagonal-rotation\'')

        super().__init__(embedding=embedding, n_components=n_components,)

        self.n_bootstraps = n_bootstraps
        self.test_case = test_case

    def _bootstrap(self, X_hat):
        t_bootstrap = np.zeros((self.n_bootstraps))
        for i in range(self.n_bootstraps):
            P = p_from_latent(X_hat)
            A1_simulated = rdpg_from_p(P)
            A2_simulated = rdpg_from_p(P)
            X1_hat_simulated, X2_hat_simulated = self._embed(A1_simulated, A2_simulated)
            t_bootstrap[i] = self._difference_norm(X1_hat_simulated, X2_hat_simulated)

        return t_bootstrap

    def _difference_norm(self, X1, X2):
        if self.embedding in ['ase', 'lse']:
            if self.test_case == 'rotation':
                R = orthogonal_procrustes(X1, X2)[0]
                return np.linalg.norm(np.dot(X1, R) - X2)
            elif self.test_case == 'scalar-rotation':
                R, s = orthogonal_procrustes(X1, X2)
                return np.linalg.norm(s / np.sum(X1**2) * np.dot(X1, R) - X2)
            elif self.test_case == 'diagonal-rotation':
                
                raise NotImplementedError()
                # normX1 = np.linalg.norm(X1, axis=1)
                # normX2 =  np.linalg.norm(X2, axis=1)
                # # print(X1)
                # # print(normX1.shape)
                # # print(normX1[:, None])
                # # X1 = np.divide(X1,normX1[:, None])
                # # X2 = np.divide(X1, normX2[:, None])
                R, s = orthogonal_procrustes(X1, X2)
                X2 = np.dot(X2, R)
                # X2 = s / np.sum(X2**2) * X2
                X2 = s * X2
                return np.linalg.norm(X1 - X2)
            elif self.test_case == 'scalar-diagonal-rotation':
                '''
                X1 = X1-np.mean(X1, 0)
                X2 = X1-np.mean(X2, 0)
                normX1 = np.linalg.norm(X1)
                normX2 =  np.linalg.norm(X2)
                X1 /= normX1
                X2 /= normX2
                R,s = orthogonal_procrustes(X1, X2)
                X2 = np.dot(X2, R.T)*s
                return np.linalg.norm(X1 - X2)
                '''
                mx1, mx2, disparity = procrustes(X1,X2)
                return disparity
        else:
            return np.linalg.norm(X1 - X2)

    def _embed(self, A1, A2):
        if self.embedding not in ['ase', 'lse', 'omnibus']:
            raise ValueError('Invalid embedding method "{}"'.format(self.embedding))

        if self.n_components is None:
            raise NotImplementedError('Wait for dimselect') # TODO

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

    def fit(self, A1, A2):
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        if not is_symmetric(A1) or not is_symmetric(A2):
            raise NotImplementedError() # TODO asymmetric case
        if A1.shape != A2.shape:
            raise ValueError('Input matrices do not have matching dimensions')
        
        X_hats = self._embed(A1, A2)
        T_sample = self._difference_norm(X_hats[0], X_hats[1])
        T1_bootstrap = self._bootstrap(X_hats[0])
        T2_bootstrap = self._bootstrap(X_hats[1])

        # Continuity correction - note that the +0.5 causes p > 1 sometimes # TODO 
        p1 = (len(T1_bootstrap[T1_bootstrap >= T_sample]) + 0.5) / self.n_bootstraps
        p2 = (len(T2_bootstrap[T2_bootstrap >= T_sample]) + 0.5) / self.n_bootstraps

        p = max(p1, p2)

        # TODO : what to store as fields here, or make _private fields
        # at least for the sake of testing, I'm going to keep everything

        self.T1_bootstrap = T1_bootstrap
        self.T2_bootstrap = T2_bootstrap
        self.T_sample = T_sample
        self.p1 = p1
        self.p2 = p2
        self.p = p

        return p
