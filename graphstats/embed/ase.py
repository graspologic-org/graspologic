
# ase.py
# Created by Ben Pedigo on 2018-09-15.
# Email: bpedigo@jhu.edu

from graphstats.embed.embed import BaseEmbed
from graphstats.utils import import_graph
from graphstats.embed.svd import selectSVD

class ASEmbed(BaseEmbed):
    """
    Class for computing the adjacency spectral embedding of a graph 
    """
    
    def __init__(self, method=selectSVD, *args, **kwargs):
        """
        Adjacency spectral embeding of a graph 

        Parameters
        ----------
            n_components: int, optional (defaults None) 
                Number of embedding dimensions. If unspecified, uses graphstats.dimselect
        """ 
        super().__init__(method=method, *args, **kwargs)
    
    def fit(self, graph):
        A = import_graph(graph)
        self._reduce_dim(A)
        return self.lpm
        

