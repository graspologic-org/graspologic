# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, Hashable, Optional, Union

import networkx as nx
import numpy as np
from beartype import beartype

from graspologic.embed import OmnibusEmbed
from graspologic.embed.base import SvdAlgorithmType
from graspologic.preconditions import check_argument, is_real_weighted
from graspologic.types import List, Set, Tuple
from graspologic.utils import (
    augment_diagonal,
    largest_connected_component,
    pass_to_ranks,
    remove_loops,
)

from . import __SVD_SOLVER_TYPES
from ._elbow import _index_of_elbow
from .embeddings import Embeddings


@beartype
def omnibus_embedding_pairwise(
    graphs: List[Union[nx.Graph, nx.OrderedGraph, nx.DiGraph, nx.OrderedDiGraph]],
    dimensions: int = 100,
    elbow_cut: Optional[int] = None,
    svd_solver_algorithm: SvdAlgorithmType = "randomized",
    svd_solver_iterations: int = 5,
    svd_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    use_laplacian: bool = False,
) -> List[Tuple[Embeddings, Embeddings]]:
    """
    Generates a pairwise omnibus embedding for each pair of graphs in a list of graphs using the adjacency matrix.
    If given graphs A, B, and C, the embeddings will be computed for A, B and B, C.

    If the node labels differ between each pair of graphs, then those nodes will only be found in the resulting embedding
    if they exist in the largest connected component of the union of all edges across all graphs in the time series.

    Graphs will always have their diagonal augmented. In other words, a self-loop
    will be created for each node with a weight corresponding to the weighted degree.

    Lastly, all weights will be rescaled based on their relative rank in the graph,
    which is beneficial in minimizing anomalous results if some edge weights are
    extremely atypical of the rest of the graph.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, nx.OrderedGraph, nx.DiGraph, nx.OrderedDiGraph]]
          A list of undirected or directed graphs. The graphs **must**:

          - be fully numerically weighted (every edge must have a real, numeric weight
            or else it will be treated as an unweighted graph)
          - be a basic graph (meaning it should not be a multigraph; if you have a
            multigraph you must first decide how you want to handle the weights of the
            edges between two nodes, whether summed, averaged, last-wins,
            maximum-weight-only, etc)
    dimensions : int (default=100)
          Dimensions to use for the svd solver.
          For undirected graphs, if ``elbow_cut==None``, you will receive an embedding
          that has ``nodes`` rows and ``dimensions`` columns.
          For directed graphs, if ``elbow_cut==None``, you will receive an embedding that
          has ``nodes`` rows and ``2*dimensions`` columns.
          If ``elbow_cut`` is specified to be not ``None``, we will cut the embedding at
          ``elbow_cut`` elbow, but the provided ``dimensions`` will be used in the
          creation of the SVD.
    elbow_cut : Optional[int] (default=None)
          Using a process described by Zhu & Ghodsi in their paper "Automatic
          dimensionality selection from the scree plot via the use of profile likelihood",
          truncate the dimensionality of the return on the ``elbow_cut``-th elbow.
          By default this value is ``None`` but can be used to reduce the dimensionality
          of the returned tensors.
    svd_solver_algorithm : str (default="randomized")
          allowed values: {'randomized', 'full', 'truncated'}

          SVD solver to use:

              - 'randomized'
                  Computes randomized svd using
                  :func:`sklearn.utils.extmath.randomized_svd`
              - 'full'
                  Computes full svd using :func:`scipy.linalg.svd`
                  Does not support ``graph`` input of type scipy.sparse.csr_matrix
              - 'truncated'
                  Computes truncated svd using :func:`scipy.sparse.linalg.svds`
    svd_solver_iterations : int (default=5)
          Number of iterations for randomized SVD solver. Not used by 'full' or
          'truncated'. The default is larger than the default in randomized_svd
          to handle sparse matrices that may have large slowly decaying spectrum.
    svd_seed : Optional[int] (default=None)
          Used to seed the PRNG used in the ``randomized`` svd solver algorithm.
    weight_attribute : str (default="weight")
          The edge dictionary key that contains the weight of the edge.
    use_laplacian : bool (default=False)
          Determine whether to use the Laplacian matrix of each graph in order to
          calculate the omnibus embedding using the Laplacian spectral embedding
          technique.

    Returns
    -------
    List[Tuple[Embeddings, Embeddings]]

    Raises
    ------
    beartype.roar.BeartypeCallHintParamViolation if parameters do not match type hints
    ValueError if values are not within appropriate ranges or allowed values

    See Also
    --------
    graspologic.pipeline.embed.Embeddings
    graspologic.embed.OmnibusEmbed
    graspologic.embed.AdjacencySpectralEmbed
    graspologic.embed.select_svd

    References
    ----------
    .. [1] Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017,
         November). A central limit theorem for an omnibus embedding of multiple random
         dot product graphs. In Data Mining Workshops (ICDMW), 2017 IEEE International
         Conference on (pp. 964-967). IEEE.

    .. [2] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
         Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
         Journal of the American Statistical Association, Vol. 107(499), 2012

    .. [3] Levin, K., Roosta-Khorasani, F., Mahoney, M. W., & Priebe, C. E. (2018).
          Out-of-sample extension of graph adjacency spectral embedding. PMLR: Proceedings
          of Machine Learning Research, 80, 2975-2984.

    .. [4] Zhu, M. and Ghodsi, A. (2006). Automatic dimensionality selection from the
          scree plot via the use of profile likelihood. Computational Statistics & Data
          Analysis, 51(2), pp.918-930.
    """
    check_argument(len(graphs) > 1, "more than one graph is required")

    check_argument(dimensions >= 1, "dimensions must be positive")

    check_argument(elbow_cut is None or elbow_cut >= 1, "elbow_cut must be positive")

    check_argument(
        svd_solver_algorithm in __SVD_SOLVER_TYPES,
        f"svd_solver_algorithm must be one of the values in {','.join(__SVD_SOLVER_TYPES)}",
    )

    check_argument(svd_solver_iterations >= 1, "svd_solver_iterations must be positive")

    check_argument(
        svd_seed is None or 0 <= svd_seed <= 2**32 - 1,
        "svd_seed must be a nonnegative, 32-bit integer",
    )

    used_weight_attribute = _graphs_precondition_checks(graphs, weight_attribute)
    perform_augment_diagonal = not use_laplacian

    graph_embeddings = []

    # create a graph that contains all nodes and edges across the entire corpus
    union_graph = graphs[0].copy()
    for graph in graphs[1:]:
        union_graph.add_edges_from(graph.edges())

    union_graph_lcc: Union[
        nx.Graph, nx.Digraph, nx.OrderedGraph, nx.OrderedDiGraph
    ] = largest_connected_component(union_graph)
    union_graph_lcc_nodes: Set[Any] = set(list(union_graph_lcc.nodes()))

    union_node_ids = np.array(list(union_graph_lcc_nodes))

    previous_graph = graphs[0].copy()

    for graph in graphs[1:]:
        current_graph = graph.copy()

        # assure both graphs contain the exact same node set
        # by removing nodes or adding isolates as needed
        _sync_nodes(previous_graph, union_graph_lcc_nodes)
        _sync_nodes(current_graph, union_graph_lcc_nodes)

        # remove self loops, run pass to ranks and diagonal augmentation
        previous_graph_augmented = _augment_graph(
            previous_graph,
            union_graph_lcc_nodes,
            used_weight_attribute,
            perform_augment_diagonal=perform_augment_diagonal,
        )
        current_graph_augmented = _augment_graph(
            current_graph,
            union_graph_lcc_nodes,
            used_weight_attribute,
            perform_augment_diagonal=perform_augment_diagonal,
        )

        model = OmnibusEmbed(
            n_components=dimensions,
            n_elbows=None,  # we will do elbow cuts
            algorithm=svd_solver_algorithm,
            n_iter=svd_solver_iterations,
            check_lcc=False,
            diag_aug=False,
            concat=False,
            svd_seed=svd_seed,
            lse=use_laplacian,
        )

        previous_embedding, current_embedding = model.fit_transform(
            graphs=[previous_graph_augmented, current_graph_augmented]
        )

        previous_embedding_cut = _elbow_cut_if_needed(
            elbow_cut, graph.is_directed(), model.singular_values_, previous_embedding
        )

        current_embedding_cut = _elbow_cut_if_needed(
            elbow_cut, graph.is_directed(), model.singular_values_, current_embedding
        )

        graph_embeddings.append(
            (
                Embeddings(union_node_ids, previous_embedding_cut),
                Embeddings(union_node_ids, current_embedding_cut),
            )
        )

    return graph_embeddings


def _graphs_precondition_checks(
    graphs: List[Union[nx.Graph, nx.OrderedGraph, nx.DiGraph, nx.OrderedDiGraph]],
    weight_attribute: str,
) -> Optional[str]:
    is_directed = graphs[0].is_directed()
    used_weight_attribute: Optional[str] = weight_attribute

    for graph in graphs:
        check_argument(
            is_directed == graph.is_directed(),
            "graphs must either be all directed or all undirected",
        )

        check_argument(
            not graph.is_multigraph(),
            "Multigraphs are not supported; you must determine how to represent at most "
            "one edge between any two nodes, and handle the corresponding weights "
            "accordingly",
        )

        if not is_real_weighted(graph, weight_attribute=weight_attribute):
            warnings.warn(
                f"Graphs with edges that do not have a real numeric weight set for every "
                f"{weight_attribute} attribute on every edge are treated as an unweighted "
                f"graph - which presumes all weights are `1.0`. If this is incorrect, "
                f"please add a '{weight_attribute}' attribute to every edge with a real, "
                f"numeric value (e.g. an integer or a float) and call this function again."
            )
            used_weight_attribute = None  # this supercedes what the user said, because
            # not all of the weights are real numbers, if they exist at all
            # this weight=1.0 treatment actually happens in nx.to_scipy_sparse_matrix()

    return used_weight_attribute


def _elbow_cut_if_needed(
    elbow_cut: Optional[int],
    is_directed: bool,
    singular_values: np.ndarray,
    embedding: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    embedding_arr: np.ndarray
    if elbow_cut is None:
        if isinstance(embedding, tuple) or is_directed:
            embedding_arr = np.concatenate(embedding, axis=1)
        else:
            embedding_arr = embedding
    else:
        column_index = _index_of_elbow(singular_values, elbow_cut)

        if isinstance(embedding, tuple) or is_directed:
            left, right = embedding
            left = left[:, :column_index]
            right = right[:, :column_index]
            embedding_arr = np.concatenate((left, right), axis=1)
        else:
            embedding_arr = embedding[:, :column_index]

    return embedding_arr


def _augment_graph(
    graph: Union[nx.Graph, nx.OrderedGraph, nx.DiGraph, nx.OrderedDiGraph],
    node_ids: Set[Hashable],
    weight_attribute: Optional[str],
    perform_augment_diagonal: bool = True,
) -> np.ndarray:
    graph_sparse = nx.to_scipy_sparse_matrix(
        graph, weight=weight_attribute, nodelist=node_ids
    )

    graphs_loops_removed: np.ndarray = remove_loops(graph_sparse)
    graphs_ranked: np.ndarray = pass_to_ranks(graphs_loops_removed)

    if perform_augment_diagonal:
        graphs_diag_augmented: np.ndarray = augment_diagonal(graphs_ranked)
        return graphs_diag_augmented

    return graphs_ranked


def _sync_nodes(
    graph_to_reduce: Union[nx.Graph, nx.OrderedGraph, nx.DiGraph, nx.OrderedDiGraph],
    set_of_valid_nodes: Set[Hashable],
) -> None:
    to_remove = []
    for n in graph_to_reduce.nodes():
        if n not in set_of_valid_nodes:
            to_remove.append(n)

    graph_to_reduce.remove_nodes_from(to_remove)

    for node in set_of_valid_nodes:
        if not graph_to_reduce.has_node(node):
            graph_to_reduce.add_node(node)
