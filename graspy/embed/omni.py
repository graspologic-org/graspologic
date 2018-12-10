# omni.py
# Created by Jaewon Chung on 2018-09-10.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .embed import BaseEmbed
from .svd import selectSVD
from ..utils import import_graph, get_lcc, is_fully_connected


def _check_valid_graphs(graphs):
    """
    Checks if all graphs in list have same shapes.

    Raises an ValueError if there are more than one shape in the input list,
    or if the list is empty or has one element.

    Parameters
    ----------
    graphs : list
        List of array-like with shapes (n_vertices, n_vertices).

    Raises
    ------
    ValueError
        If all graphs do not have same shape, or input list is empty or has 
        one element.
    """
    if len(graphs) <= 1:
        msg = "Omnibus embedding requires more than one graph."
        raise ValueError(msg)

    shapes = set(map(np.shape, graphs))

    if len(shapes) > 1:
        msg = "There are {} different sizes of graphs.".format(len(shapes))
        raise ValueError(msg)


def _get_omni_matrix(graphs):
    """
    Helper function for creating the omnibus matrix.

    Parameters
    ----------
    graphs : list
        List of array-like with shapes (n_vertices, n_vertices).

    Returns
    -------
    out : 2d-array
        Array of shape (n_vertices * n_graphs, n_vertices * n_graphs)
    """
    shape = graphs[0].shape
    n = shape[0]  # number of vertices
    m = len(graphs)  # number of graphs

    A = np.array(graphs, copy=False, ndmin=3)

    # Do some numpy broadcasting magic.
    # We do sum in 4d arrays and reduce to 2d array.
    # Super fast and efficient
    out = (A[:, :, None, :] + A.transpose(1, 0, 2)[None, :, :, :]).reshape(
        n * m, -1)

    # Averaging
    out /= 2

    return out


class OmnibusEmbed(BaseEmbed):
    r"""
    Omnibus embedding of arbitrary number of input graphs with matched vertex 
    sets.

    Given :math:`A_1, A_2, ..., A_m` a collection of (possibly weighted) adjacency 
    matrices of a collection :math:`m` undirected graphs with matched vertices. 
    Then the :math:`(mn \times mn)` omnibus matrix, :math:`M`, has the subgraph where 
    :math:`M_{ij} = \frac{1}{2}(A_i + A_j)`. The omnibus matrix is then embedded
    using adjacency spectral embedding.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full", 
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.
    n_elbows : int, optional, default: 2
        If `n_compoents=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.
    algorithm : {'full', 'truncated' (default), 'randomized'}, optional
        SVD solver to use:

        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
        - 'randomized'
            Computes randomized svd using 
            ``sklearn.utils.extmath.randomized_svd``
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or 
        'truncated'. The default is larger than the default in randomized_svd 
        to handle sparse matrices that may have large slowly decaying spectrum.

    Attributes
    ----------
    n_graphs_ : int
        Number of graphs
    n_vertices_ : int
        Number of vertices in each graph
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph. 
    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is 
        asymmetric. Estimated right latent positions of the graph. Otherwise, 
        None.
    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices.
    indices_ : array, or None
        If ``lcc`` is True, these are the indices of the vertices that were 
        kept.

    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension
    """

    def __init__(
            self,
            n_components=None,
            n_elbows=2,
            algorithm='randomized',
            n_iter=5,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
        )

    def fit(self, graphs):
        """
        Fit the model with graphs.

        Parameters
        ----------
        graphs : list of graphs, or array-like
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph. If array-like, the shape must be 
            (n_graphs, n_vertices, n_vertices)

        Returns
        -------
        self : returns an instance of self.
        """
        # Convert input to np.arrays
        graphs = [import_graph(g) for g in graphs]

        # Check if the input is valid
        _check_valid_graphs(graphs)

        # Save attributes
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]

        graphs = np.stack(graphs)

        # Check if Abar is connected
        if not is_fully_connected(graphs.mean(axis=0)):
            msg = r"""Input graphs are not fully connected. Results may not \
            be optimal. You can compute the largest connected component by \
            using ``graspy.utils.get_multigraph_union_lcc``."""
            warnings.warn(msg, UserWarning)

        # Create omni matrix
        omni_matrix = _get_omni_matrix(graphs)

        # Embed
        self._reduce_dim(omni_matrix)

        return self

    def fit_transform(self, graphs):
        """
        Fit the model with graphs and apply the embedding on graphs. 
        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graphs : list of graphs
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph.

        Returns
        -------
        out : array-like, shape (n_vertices * n_graphs, n_dimension) if input 
            graphs were symmetric. If graphs were directed, returns tuple of 
            two arrays (same shape as above) where the first corresponds to the
            left latent positions, and the right to the right latent positions
        """
        return self._fit_transform(graphs)
