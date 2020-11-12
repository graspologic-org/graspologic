# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, is_unweighted
from ..simulations import sample_edges


def _calculate_p(block):
    n_edges = np.count_nonzero(block)
    return n_edges / block.size


def _check_n_samples(n_samples):
    if not isinstance(n_samples, (int, float)):
        raise TypeError("n_samples must be a scalar value")
    if n_samples < 1:
        raise ValueError(
            "Invalid value for 'n_samples': %d . The sampling requires at "
            "least one sample." % (n_samples)
        )


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, directed=True, loops=False):
        if not isinstance(directed, bool):
            raise TypeError("`directed` must be of type bool")
        if not isinstance(loops, bool):
            raise TypeError("`loops` must be of type bool")
        self.directed = directed
        self.loops = loops

    def bic(self, graph):
        """
        Bayesian information criterion for the current model on the input graph.

        Note that this implicitly assumes the input graph is indexed like the
        fit model.

        Parameters
        ----------
        graph : np.ndarray
            Input graph

        Returns
        -------
        bic : float
            The lower the better
        """
        check_is_fitted(self, "p_mat_")
        return 2 * np.log(self.n_verts) * self._n_parameters() - 2 * self.score(graph)

    def mse(self, graph):
        """
        Compute mean square error for the current model on the input graph

        Note that this implicitly assumes the input graph is indexed like the
        fit model.

        Parameters
        ----------
        graph : np.ndarray
            Input graph

        Returns
        -------
        mse : float
            Mean square error for the model's fit P matrix
        """
        check_is_fitted(self, "p_mat_")
        return np.linalg.norm(graph - self.p_mat_) ** 2

    def score_samples(self, graph, clip=None):
        """
        Compute the weighted log probabilities for each potential edge.

        Note that this implicitly assumes the input graph is indexed like the
        fit model.

        Parameters
        ----------
        graph : np.ndarray
            Input graph. Must be same shape as model's :attr:`p_mat_` attribute

        clip : scalar or None, optional (default=None)
            Values for which to clip probability matrix, entries less than c or more
            than 1 - c are set to c or 1 - c, respectively.
            If None, values will not be clipped in the likelihood calculation, which may
            result in poorly behaved likelihoods depending on the model.

        Returns
        -------
        sample_scores : np.ndarray (size of ``graph``)
            log-likelihood per potential edge in the graph
        """
        check_is_fitted(self, "p_mat_")
        # P.ravel() <dot> graph * (1 - P.ravel()) <dot> (1 - graph)
        graph = import_graph(graph)
        if not is_unweighted(graph):
            raise ValueError("Model only implemented for unweighted graphs")
        p_mat = self.p_mat_.copy()

        if np.shape(p_mat) != np.shape(graph):
            raise ValueError("Input graph size must be the same size as P matrix")

        inds = None
        if not self.directed and self.loops:
            inds = np.triu_indices_from(graph)  # ignore lower half of graph, symmetric
        elif not self.directed and not self.loops:
            inds = np.triu_indices_from(graph, k=1)  # ignore the diagonal
        elif self.directed and not self.loops:
            xu, yu = np.triu_indices_from(graph, k=1)
            xl, yl = np.tril_indices_from(graph, k=-1)
            x = np.concatenate((xl, xu))
            y = np.concatenate((yl, yu))
            inds = (x, y)
        if inds is not None:
            p_mat = p_mat[inds]
            graph = graph[inds]

        # clip the probabilities that are degenerate
        if clip is not None:
            p_mat[p_mat < clip] = clip
            p_mat[p_mat > 1 - clip] = 1 - clip

        # TODO: use nonzero inds here will be faster
        successes = np.multiply(p_mat, graph)
        failures = np.multiply((1 - p_mat), (1 - graph))
        likelihood = successes + failures
        return np.log(likelihood)

    def score(self, graph):
        """
        Compute the average log-likelihood over each potential edge of the
        given graph.

        Note that this implicitly assumes the input graph is indexed like the
        fit model.

        Parameters
        ----------
        graph : np.ndarray
            Input graph. Must be same shape as model's :attr:`p_mat_` attribute

        Returns
        -------
        score : float
            sum of log-loglikelihoods for each potential edge in input graph
        """
        check_is_fitted(self, "p_mat_")
        return np.sum(self.score_samples(graph))

    @property
    def _pairwise(self):
        """This is for sklearn compliance."""
        return True

    @abstractmethod
    def fit(self, graph, y=None):
        """
        Calculate the parameters for the given graph model
        """
        return self

    def sample(self, n_samples=1):
        """
        Sample graphs (realizations) from the fitted model

        Can only be called after the the model has been fit

        Parameters
        ----------
        n_samples : int (default 1), optional
            The number of graphs to sample

        Returns
        -------
        graphs : np.array (n_samples, n_verts, n_verts)
            Array of sampled graphs, where the first dimension
            indexes each sample, and the other dimensions represent
            (n_verts x n_verts) adjacency matrices for the sampled graphs.

            Note that if only one sample is drawn, a (1, n_verts, n_verts)
            array will still be returned.
        """
        check_is_fitted(self, "p_mat_")
        _check_n_samples(n_samples)
        n_verts = self.p_mat_.shape[0]
        graphs = np.zeros((n_samples, n_verts, n_verts))
        p_mat = self.p_mat_.copy()
        p_mat[p_mat > 1] = 1
        p_mat[p_mat < 0] = 0
        for i in range(n_samples):
            graphs[i, :, :] = sample_edges(
                p_mat, directed=self.directed, loops=self.loops
            )
        return graphs

    @abstractmethod
    def _n_parameters(self):
        n_parameters = 1
        return n_parameters
