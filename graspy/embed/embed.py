#!/usr/bin/env python

# embed.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .svd import selectSVD
from ..utils import import_graph, is_symmetric


class BaseEmbed(BaseEstimator):
    """
    A base class for embedding a graph.

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

    See Also
    --------
    graspy.embed.selectSVD, graspy.embed.select_dimension
    """

    def __init__(
            self,
            n_components=None,
            n_elbows=2,
            algorithm='randomized',
            n_iter=5,
    ):
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.algorithm = algorithm
        self.n_iter = n_iter

    def _reduce_dim(self, A):
        """
        A function that reduces the dimensionality of an adjacency matrix
        using the desired embedding method.

        Parameters
        ----------
        A: array-like, shape (n_vertices, n_vertices)
            Adjacency matrix to embed.
        """
        U, D, V = selectSVD(
            A, n_components=self.n_components, n_elbows=self.n_elbows)

        if self.n_components is None:
            self.n_components = D.size

        self.singular_values_ = D
        self.latent_left_ = U @ np.diag(np.sqrt(D))
        if not is_symmetric(A):
            self.latent_right_ = V.T @ np.diag(np.sqrt(D))
        else:
            self.latent_right_ = None

    @abstractmethod
    def fit(self, graph):
        """
        A method for embedding.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        Returns
        -------
        lpm : LatentPosition object
            Contains X (the estimated latent positions), Y (same as X if input is
            undirected graph, or right estimated positions if directed graph), and d.

        See Also
        --------
        import_graph, LatentPosition
        """
        # call self._reduce_dim(A) from your respective embedding technique.
        # import graph(s) to an adjacency matrix using import_graph function
        # here

        return self

    def _fit_transform(self, graph):
        "Fits the model and returns the estimated latent positions"
        self.fit(graph)

        if self.latent_right_ is None:
            return self.latent_left_
        else:
            return self.latent_left_, self.latent_right_

    def fit_transform(self, graph):
        """
        Fit the model with graphs and apply the transformation. 

        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        Returns
        -------
        out : np.ndarray, shape (n_vertices, n_dimension) OR tuple (len 2)
            where both elements have shape (n_vertices, n_dimension)
            A single np.ndarray represents the latent position of an undirected
            graph, wheras a tuple represents the left and right latent positions 
            for a directed graph
        """
        return self._fit_transform(graph)