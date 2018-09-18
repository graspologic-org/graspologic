
# ase.py
# Created by Ben Pedigo on 2018-09-15.
# Email: bpedigo@jhu.edu

# Adapted from Disa Mhembere  

from graphstats.embed.embed import BaseEmbed
from graphstats.utils import import_graph

class ASEEmbed(BaseEmbed):
    """
    Class for computing the adjacency spectral embedding of a graph 
    """
    
    def __init__(self, k=2,):
        """
        Adjacency spectral embeding of a graph 

        Parameters
        ----------
            n_components: int, optional (defaults None) 
                Number of embedding dimensions. If unspecified, uses graphstats.dimselect
        """ 
        self.k = k
        super().__init__(k=k)
    
    def fit(self, graph):
        A = import_graph(graph)
        self._reduce_dim(A)
        return self.lpm
        

