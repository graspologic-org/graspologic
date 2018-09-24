# ase.py
# Created by Ben Pedigo on 2018-09-15.
# Email: bpedigo@jhu.edu

# Eric Bridgeford

from .embed import BaseEmbed
from .svd import selectSVD
from ..utils import import_graph


class AdjacencySpectralEmbed(BaseEmbed):
    """
    Class for computing the adjacency spectral embedding of a graph 
    
    The adjacency spectral embedding (ASE) is a k-dimensional Euclidean representation of 
    the graph based on its adjacency matrix [1]_. It relies on an SVD to reduce the dimensionality
    to the specified k, or if k is unspecified, can find a number of dimensions automatically
    (see graphstats.embed.svd.selectSVD).

    Parameters
    ----------
    method: object (default `selectSVD`)
        the method to use for dimensionality reduction.
    args: list, optional (default None)
        options taken by the desired embedding method as arguments.
        See graphstats.embed.svd.selectSVD for default args
    kwargs: dict, optional (default None)
        options taken by the desired embedding method as key-worded
        arguments. See graphstats.embed.svd.selectSVD for default kwargs

    See Also
    --------
    graphstats.embed.svd.selectSVD, graphstats.embed.svd.selectDim, 
    graphstats.embed.embed.BaseEmbed

    Notes
    -----
    The singular value decomposition: 

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the adjacency
    matrix of the graph. These basis vectors (in the matrices U or V) are ordered according 
    to the amount of variance they explain in the original matrix. By selecting a subset of these
    basis vectors (through our choice of dimensionality reduction) we can find a lower dimensional 
    space in which to represent the graph

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
    Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
    Journal of the American Statistical Association, Vol. 107(499), 2012

    Examples
    --------


    """

    def __init__(self, method=selectSVD, *args, **kwargs):
        super().__init__(method=method, *args, **kwargs)

    def fit(self, graph):
        """
        Fit ASE model to input graph

        Parameters
        ----------
        graph : array_like or networkx.Graph
            input graph to embed. see graphstats.utils.import_graph

        Returns
        -------
        lpm : LatentPosition object
            Contains X (the estimated latent positions), Y (same as X if input is
            undirected graph, or right estimated positions if directed graph), and d (eigenvalues
            if undirected graph, singular values if directed graph).

        See Also
        --------
        graphstats.utils.import_graph, graphstats.embed.lpm, graphstats.embed.embed

        Examples
        --------
       
        """
        A = import_graph(graph)
        self._reduce_dim(A)
        return self.lpm
