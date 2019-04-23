from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, is_almost_symmetric, binarize


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, fit_weights=False, directed=True, loops=True):
        self.fit_weights = fit_weights
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
        # return

    def score_samples(self, graph):
        """
        Compute the weighted log probabilities for each sample.
        """
        # for each edge
        # look up where you are in the sbm
        #   if dcsbm, look up where you are in the dcvector
        #   take the product of those for your indices
        # for each non edge, also neet to find 1 - p...

        # if we had the p matrix ....
        # this would be as simple as
        # where graph is the unweighted graph:
        # P.ravel() <dot> graph * (1 - P.ravel()) <dot> (1 - graph)
        bin_graph = binarize(graph)
        p_mat = self.p_mat_.copy()
        c = 1 / graph.size
        p_mat[p_mat < c] = c
        p_mat[p_mat > 1 - c] = 1 - c
        successes = np.multiply(p_mat, bin_graph)  # TODO: use nonzero inds here
        failures = np.multiply((1 - p_mat), (1 - bin_graph))
        likelihood = successes + failures
        # print(likelihood)
        # print(self.p_mat_)
        # return np.log(likelihood)
        return np.log(likelihood)

    def score(self, graph):
        """
        Compute the per-sample average log-likelihood of the given data X.
        """
        # return self.score_samples(graph).mean()
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

    @abstractmethod
    def sample(self):
        """
        sample 1 graph from the model 
        """
        graph = 1
        return graph

    @abstractmethod
    def _n_parameters(self):
        n_parameters = 1
        return n_parameters


def _calculate_p(block):
    n_edges = np.count_nonzero(block)
    return n_edges / block.size


def _fit_weights(block):
    return 1


# def _product(*arrs):


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


def bic(l_hat, n_samples, n_params):
    return np.log(n_samples) * n_params - 2 * np.log(l_hat)
