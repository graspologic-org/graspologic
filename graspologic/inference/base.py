# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from abc import abstractmethod
from sklearn.base import BaseEstimator


class BaseInference(BaseEstimator):
    """
    Base class for inference tasks such as semiparametric latent position test
    and nonparametric latent distribution test.

    Parameters
    ----------
    n_components : None (default), or int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are chosen via the Zhu and Godsi algorithm.
    """

    def __init__(self, n_components=None):
        if (not isinstance(n_components, (int, np.integer))) and (
            n_components is not None
        ):
            raise TypeError("n_components must be int or np.integer")
        if n_components is not None and n_components <= 0:
            raise ValueError(
                "Cannot embed into {} dimensions, must be greater than 0".format(
                    n_components
                )
            )
        self.n_components = n_components

    @abstractmethod
    def _embed(self, A1, A2, n_componets):
        """
        Computes the latent positions of input graphs

        Parameters
        ----------
        A1 : np.ndarray, shape (n_vertices, n_vertices)
            Adjacency matrix of the first graph
        A2 : np.ndarray, shape (n_vertices, n_vertices)
            Adjacency matrix of the second graph

        Returns
        -------
        X1_hat : array-like, shape (n_vertices, n_components)
            Estimated latent positions of the vertices in the first graph
        X2_hat : array-like, shape(n_vertices, n_components)
            Estimated latent positions of the vertices in the second graph
        """

    @abstractmethod
    def fit(self, A1, A2):
        """
        Compute the test statistic and the null distribution.

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.
            If np.ndarray, shape must be ``(n_vertices, n_vertices)`` for both
            graphs, where ``n_vertices`` is the same for both

        Returns
        ------
        self
        """
        pass

    def fit_predict(self, A1, A2):
        """
        Fits the model and returns the p-value
        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.
            If np.ndarray, shape must be ``(n_vertices, n_vertices)`` for both
            graphs, where ``n_vertices`` is the same for both

        Returns
        ------
        p_value_ : float
            The overall p value from the test
        """
        self.fit(A1, A2)
        return self.p_value_
