from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, is_almost_symmetric


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, fit_weights=False, directed=True, loops=True):
        self.fit_weights = fit_weights
        self.directed = directed
        self.loops = loops

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

