import warnings
import numpy as np
import networkx as nx
from sklearn.manifold import Isomap
from sklearn.neighbors import BallTree
from scipy.stats import ks_2samp as kstest

from .base import BaseInference
from ..utils import import_graph, is_symmetric
from ..embed import AdjacencySpectralEmbed, select_dimension


class LatentStructureTest(BaseInference):
    r"""
    Two sample hypothesis test for the problem of determining
    whether two latent structure graphs with the same latent
    curve have the same distribution of points along their curves

    Parameters
    ----------
    n_components : None (default), or int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    References
    ----------
    .. [2] Athreya, Avanti, et al. "On estimation and
       inference in latent structure random graphs."
       arXiv preprint arXiv:1806.01401, 2018
    """

    def __init__(self, n_componets):
        super().__init__(embedding="ase", n_components=n_componets)

    def _is_connected_at_k(self, points, k, tree):
        # build graph at current k
        adj = np.zeros((points.shape[0], points.shape[0]))
        for anchor_idx, anchor_point in enumerate(points):
            # +1 since the tree will return the original point
            # as nearest to itself
            dists, neighbor_idxs = tree.query(anchor_point.reshape(1, -1), k=k + 1)
            # because query wraps in a 1 dim for some reason
            dists = np.squeeze(dists)
            neighbor_idxs = np.squeeze(neighbor_idxs)
            for dist_idx, neighbor_idx in enumerate(neighbor_idxs):
                # graph is hollow and symmetric
                if neighbor_idx != anchor_idx:
                    adj[anchor_idx][neighbor_idx] = dists[dist_idx]
                    adj[neighbor_idx][anchor_idx] = dists[dist_idx]
        g = nx.from_numpy_matrix(adj)
        return nx.is_connected(g)

    def _get_min_neighbors(self, X, guess, max_iters=1000):
        tree = BallTree(X)
        k = guess
        min_true = None
        max_false = None
        for i in range(max_iters):
            if self._is_connected_at_k(X, k, tree):
                min_true = k
                if max_false is None:
                    k = k // 2
                else:
                    k = (min_true - max_false) // 2 + max_false
            else:
                max_false = k
                if min_true is None:
                    k *= 2
                else:
                    k = (min_true - max_false) // 2 + max_false
            if not min_true is None and not max_false is None:
                if min_true - max_false == 1:
                    return min_true
        raise TimeoutError(
            "Max iterations reached in LatentStructureTest._get_min_neighbors - try a different initial guess!"
        )

    def _embed(self, A1, A2, check_lcc=True):
        X1_hat = AdjacencySpectralEmbed(
            n_components=self.n_components, check_lcc=check_lcc
        ).fit_transform(A1)
        X2_hat = AdjacencySpectralEmbed(
            n_components=self.n_components, check_lcc=check_lcc
        ).fit_transform(A2)
        return (X1_hat, X2_hat)

    def _estimate_curve_positions(self, X, initial_neighbors, use_min):
        if not use_min:
            warnings.warn(
                "use_min is set to False in LatentStructureTest.fit - Nearest neighbor graph connectivity will NOT be verified!"
            )
            neighbors = initial_neighbors
        else:
            neighbors = self._get_min_neighbors(X, guess=initial_neighbors)
        manif = Isomap(n_neighbors=neighbors, n_components=1).fit_transform(X).flatten()
        standard_manif = (manif - np.min(manif)) / (np.max(manif) - np.min(manif))
        return standard_manif

    def fit(self, A1, A2, initial_neighbors=20, use_min=True):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.
            If np.ndarray, shape must be ``(n_vertices, n_vertices)`` for both graphs,
            where ``n_vertices`` is the same for both

        initial_neighbors : int
            The number of neighbors to begin the isomap kernel connectivity search at

        use_min : boolean, optional (default=True)
            Search for the min number of neighbors such that the isomap kernel is connected,
            and use this as our kernel. Otherwise, builds a kernel out of only initial_neighbors
            many neighbors of each point

        Returns
        -------
        T : float
            The test statistic corresponding to the specified hypothesis test
        """
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        if not is_symmetric(A1) or not is_symmetric(A2):
            raise NotImplementedError()  # TODO asymmetric case
        if self.n_components is None:
            # get the last elbow from ZG for each and take the maximum
            num_dims1 = select_dimension(A1)[0][-1]
            num_dims2 = select_dimension(A2)[0][-1]
            self.n_components = max(num_dims1, num_dims2)
        X_hats = self._embed(A1, A2)
        curve_positions_1 = self._estimate_curve_positions(
            X_hats[0], initial_neighbors, use_min
        )
        curve_positions_2 = self._estimate_curve_positions(
            X_hats[1], initial_neighbors, use_min
        )
        """
        The null is that the LSMs have the same densities on their
        lantent curves. If either the original or flipped KS test
        gives a large p value, then there exists a rotation of the
        latent curve that aligns the densities, which is evidence
        against the alternative. Thus, we want to use the statistic
        associated with the larger p value
        """
        stat, p = kstest(curve_positions_1, curve_positions_2)
        stat_neg, p_neg = kstest(curve_positions_1, 1.0 - curve_positions_2)
        if p > p_neg:
            T = stat
        else:
            T = stat_neg
            curve_positions_2 = curve_positions_2
        return {
            "stat": T,
            "curve_positions_1": curve_positions_1,
            "curve_positions_2": curve_positions_2,
        }
