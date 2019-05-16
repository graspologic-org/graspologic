from .base import BaseGraphEstimator
from ..embed import AdjacencySpectralEmbed
from ..simulations import rdpg, p_from_latent, sample_edges
from ..utils import import_graph, augment_diagonal
import numpy as np


class RDPGEstimator(BaseGraphEstimator):
    def __init__(
        self,
        fit_weights=False,
        fit_degrees=False,
        directed=True,
        loops=True,
        n_components=None,
    ):
        super().__init__(fit_weights=fit_weights, directed=directed, loops=loops)
        self.fit_degrees = fit_degrees
        self.n_components = n_components

    def fit(self, graph, y=None):
        # allow all ase kwargs?
        graph = import_graph(graph)
        self.n_verts = graph.shape[0]
        graph = augment_diagonal(graph, weight=1)
        graph += 1 / graph.size
        # graph[graph == 0] += 1000 * 1 / graph.size
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        latent = ase.fit_transform(graph)
        # if len(latent) == 1:
        #     latent = (latent, latent)
        self.latent_ = latent
        if type(self.latent_) == tuple:
            X = self.latent_[0]
            Y = self.latent_[1]
        else:
            X = self.latent_
            Y = None
        self.p_mat_ = p_from_latent(X, Y, rescale=False, loops=True)
        # TODO should this loops be here
        return self

    def sample(self, n_samples=1):
        # TODO: or more generally, should diagonal factor into the calculation of p for the other
        #  models
        # graph = rdpg(X, Y, loops=self.loops, directed=self.directed)
        samples = []
        for i in range(n_samples):
            graph = sample_edges(self.p_mat_, loops=self.loops, directed=self.directed)
            samples.append(graph)
        samples = np.array(samples)
        return np.squeeze(samples)

    def _n_parameters(self):
        if type(self.latent_) == tuple:
            return 2 * self.latent_[0].size
        else:
            return self.latent_.size
