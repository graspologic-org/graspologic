import numpy as np

from .base import BaseInference
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed


class SemiparamatricTest(BaseInference):
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

    def __init__(self, embedding=AdjacenctSpectralEmbed, n_components=None, *args, **kwargs):
        super().__init__(embedding=embedding, n_components=n_components, *args, **kwargs)

    def _bootstrap():
        
    def _embed(A1, A2):
        if embedding not in ['ase', 'lse', 'omnibus']: 
            raise ValueError('Invalid embedding method "{}"'.format(embedding))
        
        if embedding == 'ase':
            if n_components is None:
                X1_hat = AdjacencySpectralEmbed(method=selectSVD).fit_transform(A1).lpm.X
                X2_hat = AdjacencySpectralEmbed(method=selectSVD).fit_transform(A2).lpm.X
            else: 
                X1_hat = AdjacencySpectralEmbed(method=selectSVD).fit_transform(A1).lpm.X
                X2_hat = AdjacencySpectralEmbed(method=selectSVD).fit_transform(A2).lpm.X

    def fit(self, A1, A2):
        X1_hat, X2_hat = _embed(A1, A2):



    