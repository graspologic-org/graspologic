import warnings
from typing import Union

import numba as nb
import numpy as np
from scipy.sparse import lil_matrix

from graspologic.types import Tuple

warnings.filterwarnings("ignore")


class EdgeSwap:
    """
    Degree Preserving Edge Swaps

    This class allows for performing degree preserving edge swaps on graphs
    with fixed degree sequences.

    Attributes
    ----------
    adjacency : Union[np.ndarray, lil_matrix], shape (n_verts, n_verts)
        The initial adjacency matrix in which edge swaps are performed on it

    edge_list : np.ndarray, shape (n_verts, 2)
        The corresponding edge_list of adjacency


    References
    ----------
    .. [1] Fosdick, B. K., Larremore, D. B., Nishimura, J., & Ugander, J. (2018).
           Configuring random graph models with fixed degree sequences.
           Siam Review, 60(2), 315-355.

    .. [2] Carstens, C. J., & Horadam, K. J. (2017).
           Switching edges to randomize networks: what goes wrong and how to fix it.
           Journal of Complex Networks, 5(3), 337-351.

    .. [3] https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
    """

    def __init__(self, adjacency: Union[np.ndarray, lil_matrix]):
        self.adjacency = adjacency
        self.edge_list = self._do_setup()

    def _do_setup(self) -> np.ndarray:
        """
        Computes the edge_list from the adjancency matrix

        Returns
        -------
        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency
        """
        row_inds, col_inds = np.nonzero(self.adjacency)
        edge_list = np.stack((row_inds, col_inds)).T
        return edge_list

    @staticmethod
    @nb.jit
    def _edge_swap(
        adjacency: Union[np.ndarray, lil_matrix],
        edge_list: np.ndarray,
        seed: int = 1234,
    ) -> Tuple[Union[np.ndarray, lil_matrix], np.ndarray]:
        """
        Performs the edge swap on the adjacency matrix. If adjacency is
        np.ndarray, then nopython=True is used in numba, but if adjacency
        is lil_matrix, then forceobj=True is used in numba

        Parameters
        ----------
        adjacency : Union[np.ndarray, lil_matrix], shape (n_verts, n_verts)
            The initial adjacency matrix in which edge swaps are performed on it

        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency

        seed : int (default 1234), optional
            The seed with which we seed the process of choosing two random edges
            from the graph

        Returns
        -------
        adjacency : Union[np.ndarray, lil_matrix] shape (n_verts, n_verts)
            The adjancency matrix after an edge swap is performed on the graph

        edge_list : np.ndarray (n_verts, 2)
            The edge_list after an edge swap is perfomed on the graph
        """
        # checks if there are at 2 edges in the graph
        if len(edge_list) < 2:
            return adjacency, edge_list

        # choose two indices at random
        np.random.seed(seed)
        orig_inds = np.random.choice(len(edge_list), size=2, replace=False)

        u, v = edge_list[orig_inds[0]]
        x, y = edge_list[orig_inds[1]]

        # ensures no initial loops
        if u == v or x == y:
            return adjacency, edge_list

        # ensures no loops after swap (must be swap on 4 distinct nodes)
        if u == x or v == y:
            return adjacency, edge_list

        # save edge values
        w_uv = adjacency[u, v]
        w_xy = adjacency[x, y]
        w_ux = adjacency[u, x]
        w_vy = adjacency[v, y]

        # ensures no initial multigraphs
        if w_uv > 1 or w_xy > 1:
            return adjacency, edge_list

        # ensures no multigraphs after swap
        if w_ux >= 1 or w_vy >= 1:
            return adjacency, edge_list

        # perform the swap
        adjacency[u, v] = 0
        adjacency[x, y] = 0

        adjacency[u, x] = 1
        adjacency[v, y] = 1

        # DO EDGE LIST STUFF
        edge_list[orig_inds[0]] = [u, x]
        edge_list[orig_inds[1]] = [v, y]
        return adjacency, edge_list

    def _do_some_edge_swaps(
        self, n_swaps: int = 10
    ) -> Tuple[Union[np.ndarray, lil_matrix], np.ndarray]:
        """
        Performs a number of edge swaps on the graph

        Parameters
        ----------
        n_swaps : int (default 10), optional
            The number of edge swaps to be performed

        Returns
        -------
        self.adjacency : Union[np.ndarray, lil_matrix] shape (n_verts, n_verts)
            The adjancency matrix after a number of edge swaps are performed on the graph

        self.edge_list : np.ndarray (n_verts, 2)
            The edge_list after a number of edge swaps are perfomed on the graph
        """
        for swap in range(n_swaps):
            self.adjacency, self.edge_list = self._edge_swap(
                self.adjacency, self.edge_list
            )

        return self.adjacency, self.edge_list
