from .base import BaseGraphEstimator, _calculate_p
from .sbm import SBEstimator
from ..simulations import sbm, er_np, sample_edges
import numpy as np


class EREstimator(SBEstimator):
    def __init__(self, fit_weights=False, fit_degrees=False, directed=True, loops=True):
        super().__init__(
            fit_weights=fit_weights,
            fit_degrees=fit_degrees,
            directed=directed,
            loops=loops,
        )

    def fit(self, graph, y=None):
        er = super().fit(graph, y=np.ones(graph.shape[0]))
        self.p_ = er.block_p_[0, 0]  # TODO how to remove the block_p attribute?
        return self

    def _n_parameters(self):
        n_parameters = 0
        n_parameters += 1  # p
        if self.fit_degrees:
            n_parameters += self.n_verts
        return n_parameters

    # def sample(self):
    #     graph = er_np(self.n_verts, self.p_, directed=self.directed, loops=self.loops)
    #     return graph
