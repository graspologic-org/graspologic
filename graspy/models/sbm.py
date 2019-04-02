from .base import BaseGraphEstimator, _calculate_p, _fit_weights, cartprod
from ..utils import import_graph
import itertools
import numpy as np
from ..simulations import sbm


class SBEstimator(BaseGraphEstimator):
    def __init__(self, fit_weights=False, fit_degrees=False, directed=True, loops=True):
        super().__init__(fit_weights=fit_weights, directed=directed, loops=loops)
        self.fit_degrees = fit_degrees

    def fit(self, graph, y=None):
        """
        graph : 
        y : block assignments
        """
        if y is None:
            # _fit_block_assignments(graph)
            raise NotImplementedError("No a posteriori case yet")
            # would have to do some kind of GMM thingy here

        graph = import_graph(graph)
        # self.n_vertices = graph.shape[0]

        p = _calculate_p(graph)
        self.p_ = p

        # at this point assume block assignments are known
        self.vertex_labels_ = y

        # iterate over the blocks and calculate p
        block_labels, block_inv, block_sizes = np.unique(
            self.vertex_labels_, return_inverse=True, return_counts=True
        )

        self.block_sizes_ = block_sizes

        n_blocks = len(block_labels)
        block_inds = range(n_blocks)

        block_vert_inds = []
        for i in block_inds:
            inds = np.where(block_inv == i)[0]
            block_vert_inds.append(inds)

        block_pairs = cartprod(block_inds, block_inds)
        block_p = np.zeros((n_blocks, n_blocks))

        for p in block_pairs:
            from_block = p[0]
            to_block = p[1]
            from_inds = block_vert_inds[from_block]
            to_inds = block_vert_inds[to_block]
            block = graph[from_inds, :][:, to_inds]
            p = _calculate_p(block)
            block_p[from_block, to_block] = p
        self.block_p_ = block_p

        if self.fit_degrees:
            out_degree = graph.sum(axis=1)
            in_degree = graph.sum(axis=0)
            degree_corrections = out_degree + in_degree
            inds = np.append(inds, [degree_corrections.size])
            for i in block_inds:
                block_degrees = degree_corrections[block_vert_inds[i]]
                degree_corrections[block_vert_inds[i]] /= block_degrees.sum()
            self.degree_corrections_ = degree_corrections
        else:
            self.degree_corrections_ = None

        if self.fit_weights:
            raise NotImplementedError("no weighted case yet")
            # _fit_weights(graph)

        return self

    def sample(self):
        graph = sbm(
            self.block_sizes_,
            self.block_p_,
            loops=self.loops,
            directed=self.directed,
            dc=self.degree_corrections_,
        )
        return graph
