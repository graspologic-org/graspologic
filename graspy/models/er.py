from .base import BaseGraphEstimator, _calculate_p
from .sbm import SBEstimator, DCSBEstimator
from ..simulations import sbm, er_np, sample_edges
from ..utils import import_graph
import numpy as np


class EREstimator(SBEstimator):
    def __init__(self, directed=True, loops=False):
        super().__init__(directed=directed, loops=loops)

    def fit(self, graph, y=None):
        graph = import_graph(graph)
        er = super().fit(graph, y=np.ones(graph.shape[0]))
        self.p_ = er.block_p_[0, 0]
        delattr(self, "block_p_")
        return self

    def _n_parameters(self):
        n_parameters = 1  # p
        return n_parameters


class DCEREstimator(DCSBEstimator):
    def __init__(self, directed=True, loops=False, degree_directed=False):
        super().__init__(
            directed=directed, loops=loops, degree_directed=degree_directed
        )

    def fit(self, graph, y=None):
        dcer = super().fit(graph, y=np.ones(graph.shape[0]))
        self.p_ = dcer.block_p_[0, 0]
        delattr(self, "block_p_")
        return self

    def _n_parameters(self):
        n_parameters = 1  # p
        n_parameters += self.degree_corrections_.size
        return n_parameters
