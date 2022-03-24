from typing import Union

import numba as nb
import numpy as np
from scipy.sparse import lil_matrix

from graspologic.types import Tuple


class EdgeSwap:
    def __init__(self, adjacency: Union[np.ndarray, lil_matrix]):
        self.adjacency = adjacency

    def _do_setup(self) -> np.ndarray:
        row_inds, col_inds = np.nonzero(self.adjacency)
        edge_list = np.stack((row_inds, col_inds)).T
        return edge_list

    def _true_edge_swap(self, edge_list: np.ndarray) -> np.ndarray:
        self.adjacency, edge_list = self._edge_swap(self.adjacency, edge_list)
        return edge_list

    @staticmethod
    @nb.jit
    def _edge_swap(
        adjacency: Union[np.ndarray, lil_matrix], edge_list: np.ndarray
    ) -> Tuple[Union[np.ndarray, lil_matrix], np.ndarray]:
        # checks if there are at 2 edges in the graph
        if len(edge_list) < 2:
            # print("graph has less than two edges")
            return adjacency, edge_list

        # choose two indices at random
        np.random.seed(np.random.randint(12345))
        orig_inds = np.random.choice(len(edge_list), size=2, replace=False)

        u, v = edge_list[orig_inds[0]]
        x, y = edge_list[orig_inds[1]]

        # ensures no initial loops
        if u == v or x == y:
            # print("initial loops")
            return adjacency, edge_list

        # ensures no loops after swap (must be swap on 4 distinct nodes)
        if u == x or v == y:
            # print("loops after swap")
            return adjacency, edge_list

        # save edge values
        w_uv = adjacency[u, v]
        w_xy = adjacency[x, y]
        w_ux = adjacency[u, x]
        w_vy = adjacency[v, y]

        # ensures no initial multigraphs
        if w_uv > 1 or w_xy > 1:
            # print("initial multigraph")
            return adjacency, edge_list

        # ensures no multigraphs after swap
        if w_ux >= 1 or w_vy >= 1:
            # print("multigraph after swap")
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
        self, n_swaps: int = 1
    ) -> Tuple[Union[np.ndarray, lil_matrix], np.ndarray]:
        edge_list = self._do_setup()

        for swap in range(n_swaps):
            edge_list = self._true_edge_swap(edge_list)

        return self.adjacency, edge_list
