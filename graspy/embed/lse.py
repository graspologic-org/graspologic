# ase.py
# Created by Ben Pedigo on 2018-09-26.
# Email: bpedigo@jhu.edu


from .embed import BaseEmbed
from .svd import selectSVD
from ..utils import import_graph, to_laplace
import numpy as np

class LaplacianSpectralEmbed(BaseEmbed):
    """
    Class for computing the laplacian spectral embedding of a graph 
    
    The laplacian spectral embedding (LSE) is a k-dimensional Euclidean representation of 
    the graph based on its Laplacian matrix [1]_. It relies on an SVD to reduce the dimensionality
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

    is used to find an orthonormal basis for a matrix, which in our case is the Laplacian
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

    def __init__(self, form='DAD', method=selectSVD, *args, **kwargs):
        super().__init__(method=method, *args, **kwargs)
        self.form = form

    def fit(self, graph):
        """
        Fit LSE model to input graph

        By default, uses the Laplacian normalization of the form 
        .. math:: L = I - D^{-1/2} A D^{-1/2}

        Parameters
        ----------
        graph : array_like or networkx.Graph
            input graph to embed. see graphstats.utils.import_graph

        form : string 
            specifies the type of Laplacian normalization to use
            (currently supports 'I-DAD' only)

        Returns
        -------
        lpm : LatentPosition object
            Contains X (the estimated latent positions), Y (same as X if input is
            undirected graph, or right estimated positions if directed graph), and d (eigenvalues
            if undirected graph, singular values if directed graph).

        See Also
        --------
        graphstats.utils.import_graph, graphstats.embed.lpm, graphstats.embed.embed,
        graphstats.utils.to_laplace

        Examples
        --------
       
        """
        A = import_graph(graph)
        if not is_symmetric(A):
            raise ValueError('Laplacian spectral embedding not implemented/defined for directed graphs')
        L_norm = to_laplace(A, form=self.form)
        self._reduce_dim(L_norm)
        return self.lpm
