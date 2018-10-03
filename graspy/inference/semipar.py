import numpy as np

from ..embed import OmnibusEmbed


# TODO: Note that we can do semipar on vertex basis or
# on graph basis. I focus on graph basis here.
class Semipar():
    """
    Computes test statistic which is given by the Frobenius norm
    between each pairs of embeddings provided by omnibus embedding.
    Parameters
    ----------
    """

    def __init__(self, n_components=None):
        self.n_components_ = n_components
        self.omni = OmnibusEmbed(k=n_components)

    def fit(self, graphs):
        """
        Embeds input graphs using omnibus embedding, and computes the test
        statistic on graphs.
        Parameters
        ----------
        graphs : list of graphs
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph.
        Returns
        -------
        out : array-like, shape (n_graphs, n_graphs)
            A dissimilarity matrix based on Frobenous norms between pairs of
            graphs.
        """
        n_graphs = len(graphs)
        zhat = self.omni.fit_transform(graphs)
        n_vertices = zhat.shape[0] // n_graphs

        zhat = zhat.reshape(n_graphs, n_vertices, -1)
        out = np.zeros((n_graphs, n_graphs))

        for i in range(n_graphs):
            out[i] = np.linalg.norm(zhat - zhat[i], axis=(1, 2), ord='fro')

        return out