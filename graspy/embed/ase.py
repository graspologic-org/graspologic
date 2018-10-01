
# ase.py
# Created by Ben Pedigo on 2018-09-15.
# Email: bpedigo@jhu.edu

# Adapted from Disa Mhembere  

from embed import BaseEmbed
from utils import import_graph, check_square
from svd import SelectSVD
from sklearn.decomposition import TruncatedSVD
import numpy as np 

class ASEEmbedder(BaseEmbed):
    """
    Class for computing the adjacency spectral embedding of a graph 
    """
    
    def __init__(self, n_components=2, eig_scale=0.5):
        """
        Adjacency spectral embeding of a graph 

        Parameters
        ----------
            n_components: int, optional (defaults None) 
                Number of embedding dimensions. If unspecified, uses graphstats.dimselect
        """ 

        super.__init__(n_components=n_components, eig_scale=eig_scale)
    
    def _reduce_dim(self, A):
        
        if self.n_components == None:
            tsvd = SelectSVD() #TODO other parameters here? 
        else: 
            tsvd = TruncatedSVD(n_components = min(self.n_components, A.shape[0] - 1))
        
        tsvd.fit(A)
        eig_vectors = tsvd.components_.T
        eig_values = tsvd.singular_values_
        #X_hat = eig_vectors[:, :A.shape[1]].copy() what was the point of this in original code
        embedding = eig_vectors.dot(np.diag(eig_values**self.eig_scale))
        return embedding

    def fit(self, graph):
        A = import_graph(graph)
        check_square(A)
        self.embedding = self._reduce_dim(A)
        return self
        

