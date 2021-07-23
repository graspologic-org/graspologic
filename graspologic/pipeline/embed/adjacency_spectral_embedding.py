# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union
import warnings

import networkx as nx
import numpy as np

from ._elbow import _index_of_elbow
from .embeddings import Embeddings
from ..preconditions import (
    check_argument,
    check_argument_types,
    check_optional_argument_types,
)
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.utils import is_fully_connected, pass_to_ranks


def adjacency_spectral_embedding(
    graph: Union[nx.Graph, nx.DiGraph],
    dimensions: int = 100,
    elbow_cut: Optional[int] = None,
    svd_solver_algorithm: str = "randomized",
    svd_solver_iterations: int = 5,
    svd_seed: Optional[int] = None,
    weight_attribute: str = "weight",
) -> Embeddings:
    """
    Given a weighted directed or undirected networkx graph (*not* multigraph),
    generate an Embeddings object.

    Adjacency spectral embeddings are extremely egocentric. Further details
    can be found in the See Also section and the Reference papers listed below.

    In addition to diagonal augmentation, all graphs will have pass to ranks executed.

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        A simple, weighted graph, either undirected or directed. The graph **must** be
        fully weighted (every edge has a weight), it **must** be a simple graph (meaning
        it should not be a multigraph; if you have a multigraph you must first decide
        how you want to handle the weights of the edges between two nodes, whether
        summed, averaged, last-wins, maximum-weight-only, etc)
    dimensions : int (default=``100``)
        Dimensions to use for the svd solver.
        For undirected graphs, if ``elbow_cut==None``, you will receive an embedding
        that has ``nodes`` rows and ``dimensions`` columns.
        For directed graphs, if ``elbow_cut==None``, you will recieve an embedding that
        has ``nodes`` rows and ``2*dimensions`` columns.
        If ``elbow_cut`` is specified to be not ``None``, we will cut the embedding at
        ``elbow_cut``th elbow.
    elbow_cut : Optional[int] (default=``None``)
        An optional process where we will use the generated embedding
    elbow_cut
    svd_solver_algorithm
    svd_solver_iterations
    svd_seed
    weight_attribute

    Returns
    -------

    See Also
    --------
    graspologic.embed.AdjacencySpectralEmbed
    graspologic.embed.select_svd
    `Adjacency Spectral Embedding Tutorial
    <https://microsoft.github.io/graspologic/tutorials/embedding/AdjacencySpectralEmbed.html>`_

    Notes
    -----
    The singular value decomposition:

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the
    adjacency matrix of the graph. These basis vectors (in the matrices U or V) are
    ordered according to the amount of variance they explain in the original matrix.
    By selecting a subset of these basis vectors (through our choice of dimensionality
    reduction) we can find a lower dimensional space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012

    .. [2] Levin, K., Roosta-Khorasani, F., Mahoney, M. W., & Priebe, C. E. (2018).
        Out-of-sample extension of graph adjacency spectral embedding. PMLR: Proceedings
        of Machine Learning Research, 80, 2975-2984.

    """
    check_argument_types(dimensions, int, "dimensions must be an int")
    check_argument(dimensions >= 1, "dimensions must be positive")

    check_optional_argument_types(elbow_cut, int, "elbow_cut must be an int or None")
    check_argument(elbow_cut >= 1, "elbow_cut must be positive")

    check_argument_types(
        svd_solver_iterations, int, "svd_solver_iterations must be an int"
    )
    check_argument(svd_solver_iterations >= 1, "svd_solver_iterations must be positive")

    check_optional_argument_types(svd_seed, int, "svd_seed must be an int or None")
    check_argument(
        0 <= svd_seed <= 2 ** 32 - 1, "svd_seed must be a nonnegative, 32-bit integer"
    )

    check_argument_types(
        graph,
        (nx.Graph, nx.DiGraph),
        "graph must be of type networkx.Graph or networkx.DiGraph",
    )
    check_argument(
        not graph.is_multigraph(),
        "Multigraphs are not supported; you must determine how to represent at most one edge between any two nodes, and handle the corresponding weights accordingly",
    )

    if not nx.is_weighted(graph, weight=weight_attribute):
        warnings.warn(
            "Unweighted graphs are treated as if every edge has a weight of 1. If this is incorrect, please add a 'weight' attribute to every edge and call this function again."
        )
        # this weight=1.0 treatment actually happens in nx.to_scipy_sparse_matrix()

    graph_as_csr = nx.to_scipy_sparse_matrix(graph)

    if not is_fully_connected(graph):
        warnings.warn("More than one connected component detected")

    node_labels = np.array(list(graph.nodes()))

    graph_as_csr = pass_to_ranks(graph_as_csr)

    embedder = AdjacencySpectralEmbed(
        n_components=dimensions,
        n_elbows=None,  # in the short term, we do our own elbow finding
        algorithm=svd_solver_algorithm,
        n_iter=svd_solver_iterations,
        svd_seed=svd_seed,
        concat=False,
        diag_aug=True,
    )
    results = embedder.fit_transform(graph_as_csr)

    if elbow_cut is None:
        if graph.is_directed():
            results = np.concatenate(results, axis=1)
    else:
        column_index = _index_of_elbow(embedder.singular_values_)
        if graph.is_directed():
            left, right = results
            left = left[:, :column_index]
            right = right[:, :column_index]
            results = np.concatenate((left, right), axis=1)
        else:
            results = results[:, :column_index]

    embeddings = Embeddings(node_labels, results)
    return embeddings
