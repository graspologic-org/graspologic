#!/usr/bin/env python

# embed.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from abc import abstractmethod

import numpy as np
import networkx as nx
from sklearn.utils.validation import check_is_fitted

from .lpm import LatentPosition
from .svd import selectSVD
from ..utils import import_graph, is_symmetric


class BaseEmbed:
    """
    A base class for embedding a graph.

    Parameters
    ----------
    method: object (default selectSVD)
        the method to use for dimensionality reduction.
    args: list, optional (default None)
        options taken by the desired embedding method as arguments.
    kwargs: dict, optional (default None)
        options taken by the desired embedding method as key-worded
        arguments.

    See Also
    --------
    graphstats.embed.svd.SelectSVD, graphstats.embed.svd.selectDim
    """

    def __init__(self, method=selectSVD, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def _reduce_dim(self, A):
        """
        A function that reduces the dimensionality of an adjacency matrix
        using the desired embedding method.

        Parameters
        ----------
        A: {array-like}, shape (n_vertices, n_vertices)
            the adjacency matrix to embed.
        """
        X, Y, s = self.method(A, *self.args, **self.kwargs)
        if is_symmetric(A):
            Y = X.copy()
        self.lpm = LatentPosition(X, Y, s)

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

        return self.lpm

    def fit_transform(self, graph):
        """
        Fit the model with graphs and apply the transformation. 

        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        Returns
        -------
        out : array-like, shape (n_vertices, n_dimension)
        """
        try:
            check_is_fitted(self, ['lpm'], all_or_any=all)
        except:
            self.fit(graph)

        out = self.lpm.transform()
        return out