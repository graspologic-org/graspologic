import warnings
from asyncore import loop
from typing import Union

import networkx as nx
import numba as nb
import numpy as np
from beartype import beartype
from numba.core.errors import NumbaWarning
from scipy.sparse import SparseEfficiencyWarning, csr_matrix

from graspologic.preconditions import check_argument
from graspologic.types import Tuple
from graspologic.utils.utils import is_loopless, is_unweighted

warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore", category=SparseEfficiencyWarning)

# Code based on: https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
class EdgeSwap:
    """
    Degree Preserving Edge Swaps

    This class allows for performing degree preserving edge swaps to
    generate new networks with a fixed degree sequence.

    Attributes
    ----------
    adjacency : np.ndarray OR csr_matrix, shape (n_verts, n_verts)
        The initial adjacency matrix in which edge swaps are performed on it

    edge_list : np.ndarray, shape (n_verts, 2)
        The corresponding edge_list for the input network


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

    @beartype
    def __init__(self, adjacency: Union[np.ndarray, csr_matrix]):

        # check if graph is unweighted
        weight_check = is_unweighted(adjacency)
        check_argument(weight_check == True, "adjacency must be unweighted")

        loop_check = False
        direct_check = False
        if isinstance(adjacency, np.ndarray):
            # check if graph has loops
            loop_check = not is_loopless(adjacency)

            # check if graph is directed
            nx_graph = nx.from_numpy_array(adjacency)
            direct_check = nx.is_directed(nx_graph)

        else:
            # check if graph has loops
            for i in range(adjacency.shape[0]):
                if int(adjacency[i, i]) != 0:
                    loop_check = True
                    break

            # check if graph is directed
            nx_graph = nx.from_scipy_sparse_matrix(adjacency)
            direct_check = nx.is_directed(nx_graph)

        check_argument(loop_check == False, "adjacency cannot have loops")
        check_argument(direct_check == False, "adjacency must be undirected")
        self.adjacency = adjacency

        edge_list = self._do_setup()
        check_argument(len(edge_list) >= 2, "there must be at least 2 edges")
        self.edge_list = edge_list

    def _do_setup(self) -> np.array:
        """
        Computes the edge_list from the adjancency matrix

        Returns
        -------
        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency
        """

        if isinstance(self.adjacency, np.ndarray):
            adj_length = len(self.adjacency)

        else:
            adj_length = self.adjacency.get_shape()[0]
        edge_list = []

        for i in range(adj_length):
            for j in range(i, adj_length):
                if self.adjacency[i, j] == 1:
                    edge_list.append([i, j])
        edge_list = np.array(edge_list)
        return edge_list

    @staticmethod
    @nb.jit
    def _edge_swap(
        adjacency: Union[np.ndarray, csr_matrix], edge_list: np.array
    ) -> Tuple[Union[np.ndarray, csr_matrix], np.array]:
        """
        Performs the edge swap on the adjacency matrix. If adjacency is
        np.ndarray, then nopython=True is used in numba, but if adjacency
        is csr_matrix, then forceobj=True is used in numba

        Parameters
        ----------
        adjacency : np.ndarray OR csr_matrix, shape (n_verts, n_verts)
            The initial adjacency matrix in which edge swaps are performed on it

        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency

        seed : int (default 1234), optional
            The seed with which we seed the process of choosing two random edges
            from the graph

        Returns
        -------
        adjacency : np.ndarray OR csr_matrix, shape (n_verts, n_verts)
            The adjancency matrix after an edge swap is performed on the graph

        edge_list : np.ndarray (n_verts, 2)
            The edge_list after an edge swap is perfomed on the graph
        """

        # choose two indices at random
        orig_inds = np.random.choice(len(edge_list), size=2, replace=False)

        u, v = edge_list[orig_inds[0]]

        # two types of swap orientations for undirected graph
        if np.random.rand() < 0.5:
            x, y = edge_list[orig_inds[1]]
        else:
            y, x = edge_list[orig_inds[1]]

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
        adjacency[v, u] = 0
        adjacency[x, y] = 0
        adjacency[y, x] = 0

        adjacency[u, x] = 1
        adjacency[x, u] = 1
        adjacency[v, y] = 1
        adjacency[y, v] = 1

        # DO EDGE LIST STUFF
        edge_list[orig_inds[0]] = [u, x]
        edge_list[orig_inds[1]] = [v, y]
        return adjacency, edge_list

    def swap_edges(
        self, n_swaps: int = 1, seed: int = 1234
    ) -> Tuple[Union[np.ndarray, csr_matrix], np.array]:
        """
        Performs a number of edge swaps on the graph

        Parameters
        ----------
        n_swaps : int (default 1), optional
            The number of edge swaps to be performed

        Returns
        -------
        self.adjacency : np.ndarray OR csr.matrix, shape (n_verts, n_verts)
            The adjancency matrix after a number of edge swaps are performed on the graph

        self.edge_list : np.ndarray (n_verts, 2)
            The edge_list after a number of edge swaps are perfomed on the graph
        """
        np.random.randint(seed)
        for swap in range(n_swaps):
            self.adjacency, self.edge_list = self._edge_swap(
                self.adjacency, self.edge_list
            )

        return self.adjacency, self.edge_list
