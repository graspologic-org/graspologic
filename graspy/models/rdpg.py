from .base import BaseGraphEstimator
from ..embed import AdjacencySpectralEmbed
from ..simulations import rdpg


class RDPGEstimator(BaseGraphEstimator):
    def __init__(
        self,
        fit_weights=False,
        fit_degrees=False,
        directed=True,
        loops=True,
        n_components=None,  # or, make this ase_lws?
    ):
        super().__init__(fit_weights=fit_weights, directed=directed, loops=loops)
        self.fit_degrees = fit_degrees
        self.n_components = n_components

    def fit(self, graph, y=None):
        # allow all ase kwargs?
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        latent = ase.fit_transform(graph)
        # if len(latent) == 1:
        #     latent = (latent, latent)
        self.latent = latent

    def sample(self):
        if type(self.latent) == tuple:
            X = self.latent[0]
            Y = self.latent[1]
        else:
            X = self.latent
            Y = None
        graph = rdpg(X, Y, loops=self.loops, directed=self.directed)
        return graph
