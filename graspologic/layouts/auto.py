# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import umap
from sklearn.manifold import TSNE

from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, node2vec_embed
from ..partition import leiden
from ..preconditions import is_real_weighted
from ..preprocessing import cut_edges_by_weight, histogram_edge_weight
from ..utils import is_fully_connected, largest_connected_component
from .classes import NodePosition
from .nooverlap import remove_overlaps

logger = logging.getLogger(__name__)


def preprocess_for_layout(
    graph: nx.Graph,
    max_edges: int = 10000000,
) -> nx.Graph:
    """
    Automatically attempts to prune each graph to ``max_edges`` by removing the
    lowest weight edges. This pruning is approximate and will leave your graph with
    at most ``max_edges``, but is not guaranteed to be precisely ``max_edges``.

    In addition to pruning edges by weight, this function also finds the largest
    connected component in the graph.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph to be processed. This graph may have edges pruned if the count is
        too high and the largest connected component will be found.
    max_edges : int (default=10000000)
        The maximum number of edges to use when generating the embedding. The edges with
        the lowest weights will be pruned until at most ``max_edges`` exist. Warning:
        this pruning is approximate and more edges than are necessary may be pruned.
        Running in 32 bit environment you will most likely need to reduce this number
        or you will run out of memory.

    Returns
    -------
    nx.Graph
        The largest connected component of the graph after pruning edges if applicable.
    """
    graph = _approximate_prune(graph, max_edges)
    graph = largest_connected_component(graph)
    return graph


def embed_for_layout(
    graph: nx.Graph,
    algorithm: str = "n2v",
    validated: bool = False,
    dimensions: int = 128,
    num_walks: int = 10,
    window_size: int = 2,
    iterations: int = 3,
    n_components: Optional[int] = None,
    svd_algorithm: str = "randomized",
    random_seed: Optional[int] = None,
    embed_kwds={},
) -> Tuple[np.array, np.array]:
    """
    Generates a node2vec embedding, an adjacency spectral embedding, or a laplacian
    spectral embedding from a given graph.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph for which the embedding is generated. Only the largest connected component
        will be used to calculate the embedding.
    algorithm: str (default="n2v")
        The embedding algorithm to be used.
            - 'n2v'
                Generates embedding using :func:`~graspologic.embed.node2vec_embed`
            - 'ase'
                Generates embedding using :class:`~graspologic.embed.AdjacencySpectralEmbed`
            - 'lse'
                Generates embedding using :class:`~graspologic.embed.LaplacianSpectralEmbed`
    validated: bool (default=False)
        Whether the graph has been checked if it is fully connected in the undirected case
        or weakly connected in the directed case. If False and the input is not connected,
        the function will use its largest connected component to generate an embedding.
    dimensions: int (default=128)
        Applicable to node2vec embedding. Dimensionality of the word vectors.
    num_walks: int (default=10)
        Applicable to node2vec embedding. Number of walks per source.
    window_size: int (default=2)
        Applicable to node2vec embedding. Maximum distance between the current and predicted
        word within a sentence.
    iterations: int (default=3)
        Applicable to node2vec embedding. Number of epochs in stochastic gradient descent (SGD)
    n_components: int, optional (default=None)
        Applicable to adjacency spectral embedding or laplacian spectral embedding. Desired
        dimensionality of output data. If the SVD solver of the embedding algorithm is
        "full", ``n_components`` must be ``<= min(X.shape)``. Otherwise, ``n_components``
        must be ``< min(X.shape)``. If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.
    svd_algorithm: str (default="randomized")
        Applicable to adjacency or laplacian spectral embedding. SVD solver to use:
            - 'randomized'
                Computes randomized svd using
                :func:`sklearn.utils.extmath.randomized_svd`
            - 'full'
                Computes full svd using :func:`scipy.linalg.svd`
                Does not support ``graph`` input of type scipy.sparse.csr_matrix
            - 'truncated'
                Computes truncated svd using :func:`scipy.sparse.linalg.svds`
    random_seed: int, optional (default=None)
        Seed to be used for reproducible results. Default is None and will produce a
        random output. For adjacency or laplacian spectral embedding, only applicable
        if ``svd_algorithm="randomized``. Specifying a random state will provide
        consistent results between runs. In addition the environment variable
        ``PYTHONHASHSEED`` must be set to control hash randomization.
    embed_kwds: optional keywords
        See :func:`~graspologic.embed.node2vec_embed`,
        :class:`~graspologic.embed.AdjacencySpectralEmbed`, and
        :class:`~graspologic.embed.LaplacianSpectralEmbed` for other optional keywords.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing a matrix, with each row index corresponding to the embedding
        for each node, and a vector containing the corresponding vertex labels for each
        row in the matrix. The matrix and vector are positionally correlated.

    References
    ----------
    .. [1] Aditya Grover and Jure Leskovec  "node2vec: Scalable Feature Learning for Networks."
        Knowledge Discovery and Data Mining, 2016.
    .. [2] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    .. [3] Levin, K., Roosta-Khorasani, F., Mahoney, M. W., & Priebe, C. E. (2018).
        Out-of-sample extension of graph adjacency spectral embedding. PMLR: Proceedings
        of Machine Learning Research, 80, 2975-2984.
    .. [4] Von Luxburg, Ulrike. "A tutorial on spectral clustering," Statistics
        and computing, Vol. 17(4), pp. 395-416, 2007.
    .. [5] Rohe, Karl, Sourav Chatterjee, and Bin Yu. "Spectral clustering and
        the high-dimensional stochastic blockmodel," The Annals of Statistics,
        Vol. 39(4), pp. 1878-1915, 2011.
    """
    if validated is False:
        if is_fully_connected(graph) is False:
            graph = preprocess_for_layout(graph)
            msg = (
                "Input graph is not fully connected. Largest connected component of graph will be used to generate "
                "an embedding."
            )
            logger.warning(msg)

    start = time.time()
    if algorithm == "n2v":
        valid_n2v_kwds = [
            "num_walks",
            "walk_length",
            "return_hyperparameter",
            "inout_hyperparameter",
            "dimensions",
            "window_size",
            "iterations",
            "workers",
            "interpolate_walk_lengths_by_node_degree",
        ]
        if any(kwd not in valid_n2v_kwds for kwd in embed_kwds):
            invalid_args = ", ".join(
                [kwd for kwd in embed_kwds if kwd not in valid_n2v_kwds]
            )
            raise ValueError(f"Received invalid argument(s): {invalid_args}")
        tensors, labels = node2vec_embed(
            graph=graph,
            dimensions=dimensions,
            num_walks=num_walks,
            window_size=window_size,
            iterations=iterations,
            random_seed=random_seed,
            **embed_kwds,
        )
    else:
        if algorithm == "ase":
            valid_ase_kwds = [
                "n_elbows",
                "algorithm",
                "n_iter",
                "check_lcc",
                "diag_aug",
                "concat",
                "svd_seed",
            ]
            if any(kwd not in valid_ase_kwds for kwd in embed_kwds):
                invalid_args = ", ".join(
                    [kwd for kwd in embed_kwds if kwd not in valid_ase_kwds]
                )
                raise ValueError(f"Received invalid argument(s): {invalid_args}")
            embedder = AdjacencySpectralEmbed(
                n_components=n_components,
                algorithm=svd_algorithm,
                svd_seed=random_seed,
                concat=True,
                **embed_kwds,
            )
        elif algorithm == "lse":
            valid_lse_kwds = [
                "form",
                "n_elbows",
                "algorithm",
                "n_iter",
                "check_lcc",
                "regularizer",
                "concat",
            ]
            if any(kwd not in valid_lse_kwds for kwd in embed_kwds):
                invalid_args = ", ".join(
                    [kwd for kwd in embed_kwds if kwd not in valid_lse_kwds]
                )
                raise ValueError(f"Received invalid argument(s): {invalid_args}")
            embedder = LaplacianSpectralEmbed(
                n_components=n_components,
                algorithm=svd_algorithm,
                svd_seed=random_seed,
                concat=True,
                **embed_kwds,
            )
        else:
            raise ValueError(
                f"Algorithm must be 'n2v', 'ase', or 'lse', not {algorithm}"
            )
        labels = graph.nodes()
        tensors = embedder.fit_transform(graph)
    embedding_time = time.time() - start
    logger.info(f"embedding completed in {embedding_time} seconds")
    return tensors, labels


def reduce_dim_for_layout(
    embedding: np.ndarray,
    algorithm: str = "umap",
    metric: str = "euclidean",
    min_dist: float = 0.75,
    n_neighbors: int = 25,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_seed: Optional[int] = None,
    reduce_kwds={},
) -> np.ndarray:
    """
    Reduces the dimensionality of a given embedding to 2d space using UMAP or t-SNE.

    Parameters
    ----------
    embedding: np.ndarray
        The embedding to be reduced.
    algorithm: str (default="umap")
        The dimensionality reduction algorithm.
            - 'umap'
                Dimensionality reduction using UMAP.
            - 'tsne'
                Dimensionality reduction using t-SNE.
    metric: str (default="euclidean")
        Controls how distance is computed in the space of input data. See
        :class:`~umap.UMAP` or :class:`~sklearn.manifold.TSNE` for additional
        supported arguments.
    min_dist : float (default=0.75)
        Applicable to UMAP. The effective minimum distance between embedded points.
        Smaller values will result in a more clustered/clumped embedding where
        nearby points on the manifold are drawn closer together, while larger values
        will result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    n_neighbors : int (default=25)
        Applicable to UMAP. The size of local neighborhood (in terms of number of
        neighboring sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller values result in
        more local data being preserved.
    perplexity: int (default=30)
        Applicable to t-SNE. Related to the number of nearest neighbors that is used in
        other manifold learning algorithms. Larger datasets usually require a larger
        perplexity. Consider selecting a value between 4 and 100. Different values can
        result in significantly different results.
    n_iter : int (default=1000)
        Applicable to t-SNE. Maximum number of iterations for the optimization. We
        have found in practice that larger graphs require more iterations. We hope
        to eventually have more guidance on the number of iterations based on the
        size of the graph and the density of the edge connections.
    random_seed : int, optional (default=None)
        Seed to be used for reproducible results. Default is None and will produce
        a new random state. Specifying a random state will provide consistent results
        between runs. In addition the environment variable ``PYTHONHASHSEED`` must be
        set to control hash randomization.
    reduce_kwds: optional keywords
        See :class:`~umap.UMAP` or :class:`~sklearn.manifold.TSNE` for a description
        of optional keywords.

    Returns
    -------
    np.ndarray
        An array containing the input data reduced to 2d space.

    References
    ----------
    .. [1] McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection
        for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
    .. [2] Böhm, Jan Niklas; Berens, Philipp; Kobak, Dmitry. A Unifying Perspective
        on Neighbor Embeddings along the Attraction-Repulsion Spectrum. ArXiv e-prints
        2007.08902v1, 17 Jul 2020.
    .. [3] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data Using
        t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    """
    if algorithm == "umap":
        valid_umap_kwds = [
            "a",
            "angular_rp_forest",
            "b",
            "force_approximation_algorithm",
            "init",
            "learning_rate",
            "local_connectivity",
            "low_memory",
            "metric",
            "metric_kwds",
            "min_dist",
            "n_components",
            "n_epochs",
            "n_neighbors",
            "negative_sample_rate",
            "output_metric",
            "output_metric_kwds",
            "random_state",
            "repulsion_strength",
            "set_op_mix_ratio",
            "spread",
            "target_metric",
            "target_metric_kwds",
            "target_n_neighbors",
            "target_weight",
            "transform_queue_size",
            "transform_seed",
            "unique",
            "verbose",
        ]
        if any(kwd not in valid_umap_kwds for kwd in reduce_kwds):
            invalid_args = ", ".join(
                [kwd for kwd in reduce_kwds if kwd not in valid_umap_kwds]
            )
            raise ValueError(f"Received invalid argument(s): {invalid_args}")
        points_2d = umap.UMAP(
            n_components=2,
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_seed,
            **reduce_kwds,
        ).fit_transform(embedding)
    elif algorithm == "tsne":
        valid_tsne_kwds = [
            "n_components",
            "perplexity",
            "early_exaggeration",
            "learning_rate",
            "n_iter",
            "n_iter_without_progress",
            "min_grad_norm",
            "metric",
            "init",
            "verbose",
            "random_state",
            "method",
            "angle",
            "n_jobs",
            "square_distances",
        ]
        if any(kwd not in valid_tsne_kwds for kwd in reduce_kwds):
            invalid_args = ", ".join(
                [kwd for kwd in reduce_kwds if kwd not in valid_tsne_kwds]
            )
            raise ValueError(f"Received invalid argument(s): {invalid_args}")
        points_2d = TSNE(
            n_components=2,
            metric=metric,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_seed,
            **reduce_kwds,
        ).fit_transform(embedding)
    else:
        raise ValueError(f'Algorithm must be "umap" or "tsne", not {algorithm}.')
    return points_2d


def get_partitions(
    graph: nx.Graph,
    weight_attribute: str = "weight",
    validated: bool = False,
    random_seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Executes a global partitioning algorithm (:func:`graspologic.partition.leiden`)
    on the graph and generates a partition ID associated with each node position.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph for which the partitions are generated. The algorithm will run
        only on the largest connected component. If the graph is directed, the
        function will automatically convert it to an undirected graph by averaging
        the weight of the edges.
    weight_attribute: str (default="weight")
        The edge dictionary data attribute that holds the weight. The graph must
        be fully weighted or fully unweighted. If the edges are partially weighted,
        the function will raise an error.
    validated: bool (default=False)
        Whether the graph has been checked if it is fully connected in the undirected
        case or weakly connected in the directed case. If False and the input is not
        connected, the function will throw a warning and use its largest connected
        component to generate partitions.
    random_seed : int, optional (default=None)
        Seed to be used for reproducible results. Default is None and will produce
        a new random state. Specifying a random state will provide consistent results
        between runs. In addition the environment variable ``PYTHONHASHSEED`` must be
        set to control hash randomization.

    Returns
    -------
    Dict[str, int]
        A dictionary containing the nodes and their partition IDs generated by leiden.
        The keys in the dictionary are the string representations of the nodes.
    """
    if validated is False:
        if is_fully_connected(graph) is False:
            graph = preprocess_for_layout(graph)
            msg = (
                "Input graph is not fully connected. Largest connected component "
                "of graph will be used to generate partitions."
            )
            logger.warning(msg)

    if graph.is_directed():
        temp_graph = _to_undirected(graph, weight_attribute)
        msg = "Directed graph converted to undirected graph for community detection."
        logger.warning(msg)
        partitions = leiden(temp_graph, random_seed=random_seed)
    else:
        partitions = leiden(graph, random_seed=random_seed)
    return partitions


def get_node_positions(
    graph: nx.Graph,
    points_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    weight_attribute: str = "weight",
    validated: bool = False,
    random_seed: Optional[int] = None,
    adjust_overlaps: bool = True,
) -> List[NodePosition]:
    """
    Calculates the position and size of each node based upon their degree centrality.
    These positions and sizes can be further refined by an optional overlap removal
    phase.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        The graph for which the node positions are calculated. The function will
        run only on the largest connected component. If the graph is directed, it
        will automatically be converted to an undirected graph by averaging the
        weight of the edges.
    points_2d: np.ndarray
        The input data reduced to 2d space. The number of nodes in the graph and
        the array must be the same.
    labels: np.ndarray, optional (default=None)
        An array containing the nodes to calculate positions for. If None, the
        function will find positions and sizes of all nodes of the graph by
        default. If not None, the function will only find positions and sizes of
        the nodes specified by the input.
    weight_attribute: str (default="weight")
        The edge dictionary data attribute that holds the weight. The graph must
        be fully weighted or fully unweighted. If the edges are partially weighted,
        the function will raise an error.
    validated: bool (default=False)
        Whether the graph has been checked if it is fully connected in the undirected
        case or weakly connected in the directed case. If False and the input is not
        connected, the function will throw a warning and use its largest connected
        component to generate partitions.
    random_seed : int, optional (default=None)
        Seed to be used for reproducible results. Default is None and will produce
        a new random state. Specifying a random state will provide consistent results
        between runs. In addition the environment variable ``PYTHONHASHSEED`` must be
        set to control hash randomization.
    adjust_overlaps : bool (default=True)
        Make room for overlapping nodes while maintaining some semblance of the 2d
        spatial characteristics of each node.

    Returns
    -------
    List[NodePosition]
        A list of NodePositions for each node. The NodePosition object contains:
            - node_id
            - x coordinate
            - y coordinate
            - size
            - community
    """
    if validated is False:
        if is_fully_connected(graph) is False:
            graph = preprocess_for_layout(graph)
            msg = (
                "Input graph is not fully connected. Only the largest connected "
                "component will be used."
            )
            logger.warning(msg)

    if np.shape(graph)[0] != np.shape(points_2d)[0]:
        msg = "Input graph and input data points must have same number of nodes."
        raise ValueError(msg)

    if labels is None:
        labels = graph.nodes()

    degree = graph.degree()
    sizes = _compute_sizes(degree)
    covered_area = _covered_size(sizes)
    scaled_points = _scale_points(points_2d, covered_area)
    partitions = get_partitions(graph, weight_attribute, random_seed)
    positions = [
        NodePosition(
            node_id=key,
            x=scaled_points[index][0],
            y=scaled_points[index][1],
            size=sizes[key],
            community=partitions[key],
        )
        for index, key in enumerate(labels)
    ]
    if adjust_overlaps is True:
        positions = remove_overlaps(positions)
    return positions


def _approximate_prune(graph: nx.Graph, max_edges_to_keep: int = 1000000):
    num_edges = len(graph.edges())
    logger.info(f"num edges: {num_edges}")

    if num_edges > max_edges_to_keep:
        histogram, bins = histogram_edge_weight(graph, bin_directive=100)
        counts = 0
        bin_edge_for_maximum_weight = bins[0]
        for i, count in reversed(list(enumerate(histogram))):
            counts += count
            if counts >= max_edges_to_keep:
                bin_edge_for_maximum_weight = bins[i + 1]
                break
        threshold = bins[bin_edge_for_maximum_weight]
        graph = cut_edges_by_weight(
            graph, cut_threshold=threshold, cut_process="smaller_than_inclusive"
        )
        logger.debug(f"after cut num edges: {len(graph.edges())}")

    return graph


def _to_undirected(graph: nx.DiGraph, weight_attribute: str = "weight") -> nx.Graph:
    sym_g = nx.Graph()
    weighted = is_real_weighted(graph, weight_attribute=weight_attribute)
    for source, target, weight in graph.edges.data(weight_attribute):
        if weight is not None:
            edge_weighted = True
            if sym_g.has_edge(source, target):
                sym_g[source][target][weight_attribute] = (
                    sym_g[source][target][weight_attribute] + weight * 0.5
                )
            else:
                sym_g.add_edge(source, target)
                sym_g.edges[source, target].update({weight_attribute: weight * 0.5})
        else:
            edge_weighted = False
            sym_g.add_edge(source, target)
        if weighted != edge_weighted:
            msg = "Graph must be fully weighted or unweighted"
            raise ValueError(msg)
    return sym_g


def _find_min_max_degree(
    degrees: nx.classes.reportviews.DegreeView,
) -> Tuple[float, float]:
    min_degree = math.inf
    max_degree = -math.inf
    for _, degree in degrees:
        min_degree = min(min_degree, degree)
        max_degree = max(max_degree, degree)
    return min_degree, max_degree


def _compute_sizes(
    degrees: nx.classes.reportviews.DegreeView,
    min_size: float = 5.0,
    max_size: float = 150.0,
) -> Dict[Any, float]:
    min_degree, max_degree = _find_min_max_degree(degrees)
    sizes = {}
    for node_id, degree in degrees:
        if max_degree == min_degree:
            size = min_size
        else:
            normalized = (degree - min_degree) / (max_degree - min_degree)
            size = normalized * (max_size - min_size) + min_size
        sizes[node_id] = size
    return sizes


def _covered_size(sizes: Dict[Any, float]) -> float:
    total = sum(math.pow(value, 2) * math.pi for value in sizes.values())
    return total


def _get_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return min_x, min_y, max_x, max_y


def _center_points_on_origin(points: np.ndarray) -> np.ndarray:
    min_x, min_y, max_x, max_y = _get_bounds(points)
    x_range = max_x - min_x
    y_range = max_y - min_y
    center_x = x_range / 2 + min_x
    center_y = y_range / 2 + min_y
    points[:, 0] -= center_x
    points[:, 1] -= center_y
    return points


def _scale_to_unit_square(points: np.ndarray) -> np.ndarray:
    _, _, max_x, max_y = _get_bounds(points)
    points[:, 0] /= max_x
    points[:, 1] /= max_y
    return points


def _scale_to_new_bounds(points: np.ndarray, max_x: float, max_y: float) -> np.ndarray:
    points[:, 0] *= max_x
    points[:, 1] *= max_y
    return points


def _new_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    covered_area: float,
    target_ratio: float,
) -> Tuple[float, float, float, float]:
    range_x = max_x - min_x
    mid_x = min_x + range_x / 2
    range_y = max_y - min_y
    mid_y = min_y + range_y / 2
    range_ratio = range_x / range_y

    new_area = covered_area / target_ratio
    new_range_y = math.sqrt(new_area / range_ratio)
    new_range_x = new_area / new_range_y
    new_min_x, new_min_y, new_max_x, new_max_y = (
        mid_x - new_range_x / 2,
        mid_y - new_range_y / 2,
        mid_x + new_range_x / 2,
        mid_y + new_range_y / 2,
    )

    return new_min_x, new_min_y, new_max_x, new_max_y


def _scale_points(
    points: np.ndarray, covered_area: float, target_ratio: float = 0.14
) -> np.ndarray:
    # through lots of experiments we have found 14% to be a good ratio
    # of color to whitespace
    moved_points = _center_points_on_origin(points)
    min_x, min_y, max_x, max_y = _get_bounds(moved_points)

    _, _, new_max_x, new_max_y = _new_bounds(
        min_x, min_y, max_x, max_y, covered_area, target_ratio
    )

    scaled_points = _scale_to_unit_square(moved_points)

    final_locs = _scale_to_new_bounds(scaled_points, new_max_x, new_max_y)

    return final_locs
