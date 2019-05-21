from .base import BaseGraphEstimator
from ..embed import AdjacencySpectralEmbed
from ..simulations import rdpg, p_from_latent, sample_edges
from ..utils import import_graph, augment_diagonal, is_unweighted
import numpy as np


class RDPGEstimator(BaseGraphEstimator):
    def __init__(
        self,
        loops=False,
        n_components=None,
        ase_kws={},
        diag_aug_weight=1,
        plus_c_weight=1,
    ):
        super().__init__(loops=loops)

        if not isinstance(ase_kws, dict):
            raise TypeError("ase_kws must be a dict")
        if not isinstance(diag_aug_weight, (int, float)):
            raise TypeError("diag_aug_weight must be a scalar")
        if not isinstance(plus_c_weight, (int, float)):
            raise TypeError("plus_c_weight must be a scalar")
        if diag_aug_weight < 0:
            raise ValueError("diag_aug_weight must be at least 0")
        if plus_c_weight < 0:
            raise ValueError("plus_c_weight must be at least 0")

        self.n_components = n_components
        self.ase_kws = ase_kws
        self.diag_aug_weight = diag_aug_weight
        self.plus_c_weight = plus_c_weight

    def fit(self, graph, y=None):
        graph = import_graph(graph)
        if not is_unweighted(graph):
            raise NotImplementedError(
                "Graph model is currently only implemented for unweighted graphs."
            )
        graph = augment_diagonal(graph, weight=self.diag_aug_weight)
        graph += self.plus_c_weight / graph.size
        ase = AdjacencySpectralEmbed(n_components=self.n_components, **self.ase_kws)
        latent = ase.fit_transform(graph)
        self.latent_ = latent
        if type(self.latent_) == tuple:
            X = self.latent_[0]
            Y = self.latent_[1]
        else:
            X = self.latent_
            Y = None
        self.p_mat_ = p_from_latent(X, Y, rescale=False, loops=self.loops)

        return self

    def _n_parameters(self):
        if type(self.latent_) == tuple:
            return 2 * self.latent_[0].size
        else:
            return self.latent_.size
