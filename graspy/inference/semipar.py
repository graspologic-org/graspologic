import numpy as np

from .base import BaseInference
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from ..simulations import rdpg
from scipy.spatial import procrustes

class SemiparametricTest(BaseInference):
    """
    Two sample hypothesis test for the semiparamatric problem of determining
    whether two random dot product graphs have the same latent positions.

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
    """

    def __init__(self, embedding='ase', n_components=2, n_bootstraps=1000, *args, **kwargs):
        super().__init__(embedding=embedding, n_components=n_components, *args, **kwargs)
        self.n_bootstraps = n_bootstraps
 
    def _bootstrap(self, X_hat):
        t_bootstrap = np.zeros((self.n_bootstraps))
        for i in range(self.n_bootstraps):
            X1_hat_simulated = rdpg(X_hat) 
            X2_hat_simulated = rdpg(X_hat)
            f_norm = procrustes(X1_hat_simulated, X2_hat_simulated)[2] # TODO: swap out procrustes()[2] with other forms 
                                                                       # to test orthogonal case and arbitrary diagonal case
            t_bootstrap[i] = f_norm
        
        return t_bootstrap

    def _embed(self, A1, A2):
        if self.embedding not in ['ase', 'lse', 'omnibus']: 
            raise ValueError('Invalid embedding method "{}"'.format(self.embedding))
        
        if self.n_components is None:
            raise NotImplementedError('Wait for dimselect')

        X1_hat = np.array([]) 
        X2_hat = np.array([])
        if self.embedding == 'ase':
            X1_hat = AdjacencySpectralEmbed(k=self.n_components).fit(A1).X
            X2_hat = AdjacencySpectralEmbed(k=self.n_components).fit(A2).X
        elif self.embedding == 'lse':
            X1_hat = LaplacianSpectralEmbed(k=self.n_components).fit(A1).X
            X2_hat = LaplacianSpectralEmbed(k=self.n_components).fit(A2).X
        elif self.embedding == 'omnibus':
            X_hat_compound = OmnibusEmbed(k=self.n_components).fit([A1, A2]).X
            X1_hat = X_hat_compound[:A1.shape[0],:]
            X2_hat = X_hat_compound[A2.shape[1]:,:]
        
        return (X1_hat, X2_hat)

    def fit(self, A1, A2):
        X_hats = self._embed(A1, A2)

        # TODO : what to store as fields here 
        T1_bootstrap = self._bootstrap(X_hats[0])
        T2_bootstrap = self._bootstrap(X_hats[1])

        T_sample = procrustes(X_hats[0], X_hats[1])[2] # TODO: swap out procrustes()[2] with other forms 
                                                       # to test orthogonal case and arbitrary diagonal case

        p1 = len(T1_bootstrap[T1_bootstrap > T_sample] + 0.5) / self.n_bootstraps
        p2 = len(T2_bootstrap[T2_bootstrap > T_sample] + 0.5) / self.n_bootstraps

        p = max(p1, p2)

        return p
