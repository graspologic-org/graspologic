# base.py
# Created by Vikram Chandrashekhar on 2019-03-05.
# Email: vikramc@jhmi.edu

from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.validation import check_is_fitted

class BaseSignalSubgraph(ABC, BaseEstimator, ClassifierMixin):
    """
    Base Signal Subgraph class.
    """

    @abstractmethod
    def fit(self, graph, y=None):
        """
        A method for computing the signal subgraph.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        y : Ignored

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
        "Fits the model and returns the signal subgraph"
        self.fit(graph)

        if self.latent_right_ is None:
            return self.latent_left_
        else:
            return self.latent_left_, self.latent_right_

    def fit_transform(self, graph, y=None):
        """
        Fit the model with graphs and apply the transformation. 

        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        y : Ignored

        Returns
        -------
        out : np.ndarray, shape (n_vertices, n_dimension) OR tuple (len 2)
            where both elements have shape (n_vertices, n_dimension)
            A single np.ndarray represents the latent position of an undirected
            graph, wheras a tuple represents the left and right latent positions 
            for a directed graph
        """
        return self._fit_transform(graph)
