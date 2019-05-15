from .base import BaseGraphEstimator, _calculate_p
from .sbm import SBEstimator
from ..simulations import sbm, er_np, sample_edges
from ..utils import import_graph
import numpy as np


class EREstimator(SBEstimator):
    def __init__(self, directed=True, loops=True):
        super().__init__(directed=directed, loops=loops)

    def fit(self, graph, y=None):
        graph = import_graph(graph)
        er = super().fit(graph, y=np.ones(graph.shape[0]))
        self.p_ = er.block_p_[0, 0]  # TODO how to remove the block_p attribute?
        return self

    def _n_parameters(self):
        n_parameters = 0
        n_parameters += 1  # p
        return n_parameters
