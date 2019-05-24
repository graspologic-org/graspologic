from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, is_unweighted
from ..simulations import sample_edges


def _calculate_p(block):
    n_edges = np.count_nonzero(block)
    return n_edges / block.size


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


def bic(l_hat, n_samples, n_params):
    return np.log(n_samples) * n_params - 2 * np.log(l_hat)


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
    def __init__(self, directed=True, loops=True):
        if not isinstance(directed, bool):
            raise TypeError("`directed` must be of type bool")
        if not isinstance(loops, bool):
            raise TypeError("`loops` must be of type bool")
        self.directed = directed
        self.loops = loops

    def bic(self, graph):
        # first term should be ln(number of observations (edges)) * n_params
        # second term is 2 * ln(likelihood)
        # i.e. sum(ln(likelihood per edge))
        return 2 * np.log(self.n_verts) * self._n_parameters() - 2 * self.score(graph)

    def mse(self, graph):
        return np.linalg.norm(graph - self.p_mat_) ** 2

    def aic(self, graph):
        return 2 * self._n_parameters() - 2 * self.score(graph)

    def score_samples(self, graph):
        """
        Compute the weighted log probabilities for each sample.

        Assumes the graph is indexed like the fit model... 
        # TODO 
        """
        # P.ravel() <dot> graph * (1 - P.ravel()) <dot> (1 - graph)
        graph = import_graph(graph)
        if not is_unweighted(graph):
            raise ValueError("Model only implemented for unweighted graphs")

        p_mat = self.p_mat_.copy()

        # squish the probabilities that are degenerate
        c = 1 / graph.size
        p_mat[p_mat < c] = c
        p_mat[p_mat > 1 - c] = 1 - c
        # TODO: use nonzero inds here will be faster
        successes = np.multiply(p_mat, graph)
        failures = np.multiply((1 - p_mat), (1 - graph))
        likelihood = successes + failures
        return np.log(likelihood)

    def score(self, graph):
        """
        Compute the per-sample average log-likelihood of the given data X.
        """
        return np.sum(self.score_samples(graph))

    @property
    def _pairwise(self):
        """This is for sklearn compliance."""
        return True

    @abstractmethod
    def fit(self, graph, y=None):
        """
        calculate the parameters for the given graph model 
        """

        return self

    def sample(self, n_samples=1):
        """
        sample 1 graph from the model 
        """
        check_is_fitted(self, "p_mat_")
        _check_n_samples(n_samples)
        n_verts = self.p_mat_.shape[0]
        graphs = np.zeros((n_samples, n_verts, n_verts))
        for i in range(n_samples):
            graphs[i, :, :] = sample_edges(
                self.p_mat_, directed=self.directed, loops=self.loops
            )
        return graphs

    @abstractmethod
    def _n_parameters(self):
        n_parameters = 1
        return n_parameters
