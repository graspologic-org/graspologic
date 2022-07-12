from typing import Optional

import numba as nb
import numpy as np
from beartype import beartype
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils import check_scalar

from graspologic.preconditions import check_argument
from graspologic.types import AdjacencyMatrix, Tuple
from graspologic.utils import import_graph, is_loopless, is_symmetric, is_unweighted


# Code based on: https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
class EdgeSwapper:
    """
    Degree Preserving Edge Swaps

    This class allows for performing degree preserving edge swaps to
    generate new networks with the same degree sequence as the input network.

    Attributes
    ----------
    adjacency : np.ndarray OR csr_matrix, shape (n_verts, n_verts)
        The initial adjacency matrix to perform edge swaps on. Must be unweighted and undirected.

    edge_list : np.ndarray, shape (n_verts, 2)
        The corresponding edgelist for the input network

    seed: int, optional
        Random seed to make outputs reproducible, must be positive


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
    def __init__(self, adjacency: AdjacencyMatrix, seed: Optional[int] = None):

        weight_check = is_unweighted(adjacency)
        check_argument(weight_check, "adjacency must be unweighted")

        loop_check = is_loopless(adjacency)
        check_argument(loop_check, "adjacency cannot have loops")

        direct_check = is_symmetric(adjacency)
        check_argument(direct_check, "adjacency must be undirected")

        max_seed = np.iinfo(np.uint32).max
        if seed is None:
            seed = np.random.randint(max_seed, dtype=np.int64)
        seed = check_scalar(
            seed, "seed", (int, np.integer), min_val=0, max_val=max_seed
        )
        self._rng = np.random.default_rng(seed)

        adjacency = import_graph(adjacency, copy=True)

        if isinstance(adjacency, csr_matrix):
            # more efficient for manipulations which change sparsity structure
            adjacency = lil_matrix(adjacency)
            self._edge_swap_function = _edge_swap
        else:
            # for numpy input, use numba for JIT compilation
            # NOTE: not convinced numba is helping much here, look into optimizing
            self._edge_swap_function = _edge_swap_numba

        self.adjacency = adjacency

        edge_list = self._do_setup()
        check_argument(len(edge_list) >= 2, "there must be at least 2 edges")
        self.edge_list = edge_list

    def _do_setup(self) -> np.ndarray:
        """
        Computes the edge_list from the adjancency matrix

        Returns
        -------
        edge_list : np.ndarray, shape (n_verts, 2)
            The corresponding edge_list of adjacency
        """

        # get edges for upper triangle of undirected graph
        row_inds, col_inds = np.nonzero(self.adjacency)
        upper = row_inds < col_inds
        row_inds = row_inds[upper]
        col_inds = col_inds[upper]
        edge_list = np.stack((row_inds, col_inds)).T
        return edge_list

    def swap_edges(self, n_swaps: int = 1) -> Tuple[AdjacencyMatrix, np.ndarray]:
        """
        Performs a number of edge swaps on the graph

        Parameters
        ----------
        n_swaps : int (default 1), optional
            The number of edge swaps to be performed

        Returns
        -------
        adjacency : np.ndarray OR csr.matrix, shape (n_verts, n_verts)
            The adjancency matrix after a number of edge swaps are performed on the graph

        edge_list : np.ndarray (n_verts, 2)
            The edge_list after a number of edge swaps are perfomed on the graph
        """

        # Note: for some reason could not get reproducibility w/o setting seed
        # inside of the _edge_swap_function itself
        max_seed = np.iinfo(np.int32).max
        for _ in range(n_swaps):
            self.adjacency, self.edge_list = self._edge_swap_function(
                self.adjacency,
                self.edge_list,
                seed=self._rng.integers(max_seed),
            )

        adjacency = self.adjacency
        if isinstance(adjacency, lil_matrix):
            adjacency = csr_matrix(adjacency)
        else:
            adjacency = adjacency.copy()

        return adjacency, self.edge_list.copy()


def _edge_swap(
    adjacency: AdjacencyMatrix, edge_list: np.ndarray, seed: Optional[int] = None
) -> Tuple[AdjacencyMatrix, np.ndarray]:
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

    seed: int, optional
        Random seed to make outputs reproducible, must be positive

    Returns
    -------
    adjacency : np.ndarray OR csr_matrix, shape (n_verts, n_verts)
        The adjancency matrix after an edge swap is performed on the graph

    edge_list : np.ndarray (n_verts, 2)
        The edge_list after an edge swap is perfomed on the graph
    """

    # need to use np.random here instead of the generator for numba compatibility
    if seed is not None:
        np.random.seed(seed)

    # choose two indices at random
    # NOTE: using np.random here for current numba compatibility
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
    w_ux = adjacency[u, x]
    w_vy = adjacency[v, y]

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

    # update edge list
    edge_list[orig_inds[0]] = [u, x]
    edge_list[orig_inds[1]] = [v, y]
    return adjacency, edge_list


_edge_swap_numba = nb.jit(_edge_swap)
