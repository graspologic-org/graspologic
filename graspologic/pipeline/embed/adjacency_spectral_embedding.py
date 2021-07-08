# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union
import warnings

import networkx as nx
import numpy as np

from .embeddings import Embeddings
from ..assertions import assert_simple_nx_graph, assert_weighted
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.utils import is_fully_connected, pass_to_ranks


def adjacency_spectral_embedding(
    graph: Union[nx.Graph, nx.DiGraph],
    fixed_dimensionality: Optional[int] = None,
    elbow_cut: Optional[int] = 2,
    svd_solver_algorithm: str = "randomized",
    svd_solver_iterations: int = 5,
    svd_seed: Optional[int] = None,
    weight_attribute: str = "weight",
) -> Embeddings:
    assert_simple_nx_graph(
        graph,
        "Graph provided must be of type networkx.Graph or networkx.DiGraph. If you "
        "have a multigraph, you need to figure out how to represent at most one edge "
        "between any two nodes, and handle the weights accordingly (e.g. last weight "
        "wins, sum all weights, average weights, etc)",
    )
    assert_weighted(graph, weight_attribute=weight_attribute)
    is_directed = graph.is_directed()

    graph_as_csr = nx.to_scipy_sparse_matrix(graph)

    if not is_fully_connected(graph):
        warnings.warn("More than one connected component detected")

    node_labels = np.array(list(graph.nodes()))
    graph_as_csr = pass_to_ranks(graph_as_csr)

    embedder = AdjacencySpectralEmbed(
        n_components=fixed_dimensionality,
        n_elbows=elbow_cut,
        algorithm=svd_solver_algorithm,
        n_iter=svd_solver_iterations,
        svd_seed=svd_seed,  # have to expose in ase/lse/base
        concat=is_directed,
        diag_aug=True,
    )
    results = embedder.fit_transform(graph_as_csr)
    embeddings = Embeddings(node_labels, results)
    return embeddings
