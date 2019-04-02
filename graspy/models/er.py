from .base import BaseGraphEstimator, _calculate_p
from .sbm import SBEstimator
from ..simulations import sbm, er_np
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
        self.p_ = er.block_p_[0, 0]
        # self.degree_corrections_ = er.degree_corrections_
        self.n_verts = graph.shape[0]
        return self

    def sample(self):
        graph = er_np(
            self.n_verts,
            self.p_,
            directed=self.directed,
            loops=self.loops,
            dc=self.degree_corrections_,
        )
        return graph
