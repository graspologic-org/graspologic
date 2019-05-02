from .base import BaseGraphEstimator, _calculate_p, _fit_weights, cartprod
from ..utils import (
    import_graph,
    binarize,
    is_almost_symmetric,
    augment_diagonal,
    is_unweighted,
)
import numpy as np
from ..simulations import sbm, sample_edges
from ..cluster import GaussianCluster
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from sklearn.mixture import GaussianMixture


def _get_block_indices(y):
    """
    y is a length n_verts vector of labels
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )
    # self.block_sizes_ = block_sizes

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    # block_members = []
    # for bs, bl in zip(block_sizes, block_labels):
    #     block_members = block_members + bs * [bl]
    # block_members = np.array(block_members)
    # self.block_members_ = np.array(block_members)  # TODO necessary?

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_vert_inds, block_inds, block_inv


def _calculate_block_p(graph, block_inds, block_vert_inds):
    n_blocks = len(block_inds)
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
    return block_p


def _block_to_full(block_mat, inverse, shape):
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


class SBEstimator(BaseGraphEstimator):
    def __init__(
        self,
        fit_weights=False,
        fit_degrees=False,
        degree_directed=False,
        directed=True,
        loops=True,
        n_components=None,
        min_comm=1,
        max_comm=6,  # TODO some more intelligent default here?
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(fit_weights=fit_weights, directed=directed, loops=loops)
        self.fit_degrees = fit_degrees
        self.degree_directed = degree_directed
        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = {}

    def _estimate_assignments(self, graph):
        embed_graph = augment_diagonal(graph)  # TODO always?
        # TODO other regularizations
        # TODO regularized laplacians for finding communities?
        latent = AdjacencySpectralEmbed(
            n_components=self.n_components, **self.embed_kws
        ).fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        vertex_assignments = gc.fit_predict(latent)
        self.vertex_assignments_ = vertex_assignments

    def fit(self, graph, y=None):
        """
        graph : 
        y : block assignments
        """
        graph = import_graph(graph)
        if not is_unweighted(graph):
            raise NotImplementedError(
                "Graph model is currently only implemented" + " for unweighted graphs."
            )
        # self.n_verts = graph.shape[0]

        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)

        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)
        self.block_p_ = block_p

        p_mat = _block_to_full(block_p, block_inv, graph.shape)

        self.p_mat_ = p_mat

        if self.fit_weights:
            # TODO: something
            raise NotImplementedError(
                "Graph model is currently only implemented" + " for unweighted graphs."
            )

        return self

    def sample(self):
        # graph = sbm(
        #     self.block_sizes_,
        #     self.block_p_,
        #     loops=self.loops,
        #     directed=self.directed,
        #     dc=self.degree_corrections_,
        # )

        # TODO at some point we may want to sample probabilistically here for the
        # block memberships
        graph = sample_edges(self.p_mat_, directed=self.directed, loops=self.loops)
        return graph

    def _n_parameters(self):
        n_parameters = 0
        n_parameters += self.block_p_.size  # elements in block p matrix
        n_parameters += self.block_sizes_.size  # TODO - 1?
        return n_parameters


class DCSBEstimator(BaseGraphEstimator):
    def __init__(
        self,
        degree_directed=False,
        directed=True,
        loops=True,
        n_components=None,
        min_comm=1,
        max_comm=6,  # TODO some more intelligent default here?
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(directed=directed, loops=loops)
        self.degree_directed = degree_directed
        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = {}

    def _estimate_assignments(self, graph):
        # TODO
        # do regularized laplacian embedding
        #
        lse = LaplacianSpectralEmbed(form="R-DAD", **self.embed_kws)
        latent = lse.fit_transform(graph)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        self.vertex_assignments_ = gc.fit_predict(latent)

    def fit(self, graph, y=None):
        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)
        # n_blocks = len(block_inds)
        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        out_degree = np.count_nonzero(graph, axis=1).astype(float)
        in_degree = np.count_nonzero(graph, axis=0).astype(float)
        if self.degree_directed:
            degree_corrections = np.stack((out_degree, in_degree), axis=1)
        else:
            degree_corrections = out_degree + in_degree
            # new axis just so we can index later
            degree_corrections = degree_corrections[:, np.newaxis]
        for i in block_inds:
            block_degrees = degree_corrections[block_vert_inds[i]]
            degree_divisor = np.sum(block_degrees, axis=0)  # was mean
            if not isinstance(degree_divisor, np.float64):
                degree_divisor[degree_divisor == 0] = 1
            degree_corrections[block_vert_inds[i]] = (
                degree_corrections[block_vert_inds[i]] / degree_divisor
            )
        self.degree_corrections_ = degree_corrections

        delta_block_mat = _calculate_block_p(graph, block_inds, block_vert_inds)
        block_p = block_p / delta_block_mat
        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        p_mat = p_mat * np.outer(degree_corrections[:, 0], degree_corrections[:, -1])
        # p_mat[p_mat > 1] = 1
        self.p_mat_ = p_mat
        pass

    def sample(self):
        pass

    def _n_parameters(self):
        n_parameters = 0
        n_parameters += self.block_p_.size  # elements in block p matrix
        # n_parameters += self.block_sizes_.size
        n_parameters += (
            self.degree_corrections_.size + self.degree_corrections_.shape[0]
        )
        # TODO: more than this? becasue now the position of verts w/in block matters
        return n_parameters
