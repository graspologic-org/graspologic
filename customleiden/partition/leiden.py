# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, NamedTuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import math
from collections import defaultdict
import graspologic_native as gn
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec

import scipy
from beartype import beartype

from customleiden.types import AdjacencyMatrix, Dict, GraphRepresentation, List, Tuple

from .. import utils
from ..preconditions import check_argument


class _IdentityMapper:
    def __init__(self) -> None:
        self._inner_mapping: Dict[str, Any] = {}

    def __call__(self, value: Any) -> str:
        as_str = str(value)
        mapped = self._inner_mapping.get(as_str, value)
        if mapped != value:
            # we could conceivably address this by also using the hashcode of the value and
            # storing submaps but that is not super likely to occur
            raise ValueError(
                "str(value) results in a collision between distinct values"
            )
        self._inner_mapping[as_str] = mapped
        return as_str

    def original(self, as_str: str) -> Any:
        return self._inner_mapping[as_str]

    def __len__(self) -> int:
        return len(self._inner_mapping)


@beartype
def _nx_to_edge_list(
    graph: nx.Graph,
    identifier: _IdentityMapper,
    is_weighted: Optional[bool],
    weight_attribute: str,
    weight_default: float,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    check_argument(
        isinstance(graph, nx.Graph)
        and not (graph.is_directed() or graph.is_multigraph()),
        "Only undirected non-multi-graph networkx graphs are supported",
    )
    native_safe: List[Tuple[str, str, float]] = []
    edge_iter = (
        graph.edges(data=weight_attribute)
        if is_weighted is True
        else graph.edges(data=weight_attribute, default=weight_default)
    )
    for source, target, weight in edge_iter:
        source_str = identifier(source)
        target_str = identifier(target)
        native_safe.append((source_str, target_str, float(weight)))
    return graph.number_of_nodes(), native_safe


@beartype
def _adjacency_matrix_to_edge_list(
    matrix: AdjacencyMatrix,
    identifier: _IdentityMapper,
    check_directed: Optional[bool],
    is_weighted: Optional[bool],
    weight_default: float,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    check_argument(
        check_directed is True and utils.is_almost_symmetric(matrix),
        "leiden only supports undirected graphs and the adjacency matrix provided "
        "was found to be directed",
    )
    shape = matrix.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            "graphs of type np.ndarray or csr.sparse.csr.csr_array should be "
            "adjacency matrices with n x n shape"
        )

    if is_weighted is None:
        is_weighted = not utils.is_unweighted(matrix)

    native_safe: List[Tuple[str, str, float]] = []
    if isinstance(matrix, np.ndarray):
        for i in range(0, shape[0]):
            source = identifier(i)
            for j in range(i, shape[1]):
                target = identifier(j)
                weight = matrix[i][j]
                if weight != 0:
                    if not is_weighted and weight == 1:
                        weight = weight_default
                    native_safe.append((source, target, float(weight)))
    else:
        rows, columns = matrix.nonzero()
        for i in range(0, len(rows)):
            source = rows[i]
            source_str = identifier(source)
            target = columns[i]
            target_str = identifier(target)
            weight = float(matrix[source, target])
            if source <= target:
                native_safe.append((source_str, target_str, weight))

    return shape[0], native_safe


@beartype
def _edge_list_to_edge_list(
    edges: List[Tuple[Any, Any, Union[int, float]]], identifier: _IdentityMapper
) -> Tuple[int, List[Tuple[str, str, float]]]:
    native_safe: List[Tuple[str, str, float]] = []
    temp_node_set = set()

    for source, target, weight in edges:
        source_str = identifier(source)
        target_str = identifier(target)
        weight_as_float = float(weight)
        if source_str != target_str:
            native_safe.append((source_str, target_str, weight_as_float))
            temp_node_set.add(source_str)
            temp_node_set.add(target_str)
    return len(temp_node_set), native_safe


@beartype
def _community_python_to_native(
    starting_communities: Optional[Dict[Any, int]], identity: _IdentityMapper
) -> Optional[Dict[str, int]]:
    if starting_communities is None:
        return None
    native_safe: Dict[str, int] = {}
    for node_id, partition in starting_communities.items():
        node_id_as_str = identity(node_id)
        native_safe[node_id_as_str] = partition
    return native_safe


@beartype
def _community_native_to_python(
    communities: Dict[str, int], identity: _IdentityMapper
) -> Dict[Any, int]:
    return {
        identity.original(node_id_as_str): partition
        for node_id_as_str, partition in communities.items()
    }


@beartype
def _validate_common_arguments(
    extra_forced_iterations: int = 0,
    resolution: Union[float, int] = 1.0,
    randomness: Union[float, int] = 0.001,
    random_seed: Optional[int] = None,
) -> None:
    check_argument(
        extra_forced_iterations >= 0,
        "extra_forced_iterations must be a non negative integer",
    )
    check_argument(resolution > 0, "resolution must be a positive float")
    check_argument(randomness > 0, "randomness must be a positive float")
    check_argument(
        random_seed is None or random_seed > 0,
        "random_seed must be a positive integer (the native PRNG implementation is"
        " an unsigned 64 bit integer)",
    )

@beartype
def leiden(
    graph: Union[
        List[Tuple[Any, Any, Union[int, float]]],
        GraphRepresentation,
    ],
    starting_communities: Optional[Dict[Any, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: Union[int, float] = 1.0,
    randomness: Union[int, float] = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: Union[int, float] = 1.0,
    check_directed: bool = True,
    trials: int = 1,
    track_misalignment: bool = False,
    embedding_method: Optional[str] = "node2vec",
) -> Union[Dict[Any, int], Tuple[Dict[Any, int], Dict[Any, int]]]:
    """
    Leiden is a global network partitioning algorithm. Given a graph, it will iterate
    through the network node by node, and test for an improvement in our quality
    maximization function by speculatively joining partitions of each neighboring node.

    This process continues until no moves are made that increases the partitioning
    quality.

    Parameters
    ----------
    graph : Union[List[Tuple[Any, Any, Union[int, float]]], GraphRepresentation]
        A graph representation, whether a weighted edge list referencing an undirected
        graph, an undirected networkx graph, or an undirected adjacency matrix in either
        numpy.ndarray or scipy.sparse.csr_array form. Please see the Notes section
        regarding node ids used.
    starting_communities : Optional[Dict[Any, int]]
        Default is ``None``. An optional community mapping dictionary that contains a node
        id mapping to the community it belongs to. Please see the Notes section regarding
        node ids used.

        If no community map is provided, the default behavior is to create a node
        community identity map, where every node is in their own community.
    extra_forced_iterations : int
        Default is ``0``. Leiden will run until a maximum quality score has been found
        for the node clustering and no nodes are moved to a new cluster in another
        iteration. As there is an element of randomness to the Leiden algorithm, it is
        sometimes useful to set ``extra_forced_iterations`` to a number larger than 0
        where the process is forced to attempt further refinement.
    resolution : Union[int, float]
        Default is ``1.0``. Higher resolution values lead to more communities and lower
        resolution values leads to fewer communities. Must be greater than 0.
    randomness : Union[int, float]
        Default is ``0.001``. The larger the randomness value, the more exploration of
        the partition space is possible. This is a major difference from the Louvain
        algorithm, which is purely greedy in the partition exploration.
    use_modularity : bool
        Default is ``True``. If ``False``, will use a Constant Potts Model (CPM).
    random_seed : Optional[int]
        Default is ``None``. Can provide an optional seed to the PRNG used in Leiden for
        deterministic output.
    weight_attribute : str
        Default is ``weight``. Only used when creating a weighed edge list of tuples
        when the source graph is a networkx graph. This attribute corresponds to the
        edge data dict key.
    is_weighted : Optional[bool]
        Default is ``None``. Only used when creating a weighted edge list of tuples
        when the source graph is an adjacency matrix. The
        :func:`customleiden.utils.is_unweighted` function will scan these
        matrices and attempt to determine whether it is weighted or not. This flag can
        short circuit this test and the values in the adjacency matrix will be treated
        as weights.
    weight_default : Union[int, float]
        Default is ``1.0``. If the graph is a networkx graph and the graph does not have
        a fully weighted sequence of edges, this default will be used. If the adjacency
        matrix is found or specified to be unweighted, this weight_default will be used
        for every edge.
    check_directed : bool
        Default is ``True``. If the graph is an adjacency matrix, we will attempt to
        ascertain whether it is directed or undirected. As our leiden implementation is
        only known to work with an undirected graph, this function will raise an error
        if it is found to be a directed graph. If you know it is undirected and wish to
        avoid this scan, you can set this value to ``False`` and only the lower triangle
        of the adjacency matrix will be used to generate the weighted edge list.
    trials : int
        Default is ``1``. Runs leiden ``trials`` times, keeping the best partitioning
        as judged by the quality maximization function (default: modularity, see
        ``use_modularity`` parameter for details). This differs from
        ``extra_forced_iterations`` by starting over from scratch each for each trial,
        while ``extra_forced_iterations`` attempts to make microscopic adjustments from
        the "final" state.

    Returns
    -------
    Dict[Any, int]
        The results of running leiden over the provided graph, a dictionary containing
        mappings of node -> community id. Isolate nodes in the input graph are not returned
        in the result.

    Raises
    ------
    ValueError
    TypeError
    BeartypeCallHintParamViolation

    See Also
    --------
    customleiden.utils.is_unweighted

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
         guaranteeing well-connected communities", Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/graspologic-org/graspologic-native

    Notes
    -----
    No two different nodes are allowed to encode to the **same** str representation,
    e.g. node_a id of ``"1"`` and node_b id of ``1`` are different object types
    but str(node_a) == str(node_b). This collision will result in a ``ValueError``

    This function is implemented in the `graspologic-native` Python module, a module
    written in Rust for Python.
    """
    _validate_common_arguments(
        extra_forced_iterations,
        resolution,
        randomness,
        random_seed,
    )
    check_argument(trials >= 1, "Trials must be a positive integer")

    identifier = _IdentityMapper()
    node_count: int
    edges: List[Tuple[str, str, float]]
    if isinstance(graph, nx.Graph):
        node_count, edges = _nx_to_edge_list(
            graph, identifier, is_weighted, weight_attribute, weight_default
        )
    elif isinstance(graph, list):
        node_count, edges = _edge_list_to_edge_list(graph, identifier)
    else:
        node_count, edges = _adjacency_matrix_to_edge_list(
            graph, identifier, check_directed, is_weighted, weight_default
        )

    native_friendly_communities = _community_python_to_native(
        starting_communities, identifier
    )

    _quality, native_partitions = gn.leiden(
        edges=edges,
        starting_communities=native_friendly_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        seed=random_seed,
        trials=trials,
    )

    proper_partitions = _community_native_to_python(native_partitions, identifier)

    if len(proper_partitions) < node_count:
        warnings.warn(
            "Leiden partitions do not contain all nodes from the input graph because input graph "
            "contained isolate nodes."
        )

    if track_misalignment:
        if not isinstance(graph, nx.Graph):
            raise ValueError("Semantic misalignment tracking requires a networkx graph")
        if embedding_method == "node2vec":
            embeddings = compute_node2vec_embeddings(graph)
        else:
            raise NotImplementedError(f"Embedding method {embedding_method} not implemented")
        misaligned_nodes = detect_misalignment(proper_partitions, embeddings)
        return proper_partitions, misaligned_nodes

    return proper_partitions


class HierarchicalCluster(NamedTuple):
    node: Any
    """Node id"""
    cluster: int
    """Leiden cluster id"""
    parent_cluster: Optional[int]
    """Only used when level != 0, but will indicate the previous cluster id that this node was in"""
    level: int
    """
    Each time a community has a higher population than we would like, we create a subnetwork
    of that community and process it again to break it into smaller chunks. Each time we
    detect this, the level increases by 1
    """
    is_final_cluster: bool
    """
    Whether this is the terminal cluster in the hierarchical leiden process or not
    """


class HierarchicalClusters(List[HierarchicalCluster]):
    """
    HierarchicalClusters is a subclass of Python's :class:`list` class with two
    helper methods for retrieving dictionary views of the first and final
    level of hierarchical clustering in dictionary form.  The rest of the
    HierarchicalCluster entries in this list can be seen as a transition
    state log of our :func:`customleiden.partition.hierarchical_leiden` process
    as it continuously tries to break down communities over a certain size,
    with the two helper methods on this list providing you the starting point
    community map and ending point community map.
    """

    def first_level_hierarchical_clustering(self) -> Dict[Any, int]:
        """
        Returns
        -------
        Dict[Any, int]
            The initial leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in self if entry.level == 0}

    def final_level_hierarchical_clustering(self) -> Dict[Any, int]:
        """
        Returns
        -------
        Dict[Any, int]
            The last leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in self if entry.is_final_cluster}


def _from_native(
    native_cluster: gn.HierarchicalCluster,
    identifier: _IdentityMapper,
) -> HierarchicalCluster:
    if not isinstance(native_cluster, gn.HierarchicalCluster):
        raise TypeError(
            "This class method is only valid for graspologic_native.HierarchicalCluster"
        )
    node_id: Any = identifier.original(native_cluster.node)
    return HierarchicalCluster(
        node=node_id,
        cluster=native_cluster.cluster,
        parent_cluster=native_cluster.parent_cluster,
        level=native_cluster.level,
        is_final_cluster=native_cluster.is_final_cluster,
    )


@beartype
def hierarchical_leiden(
    graph: Union[
        List[Tuple[Any, Any, Union[int, float]]],
        nx.Graph,
        np.ndarray,
        scipy.sparse.csr_array,
    ],
    max_cluster_size: int = 1000,
    starting_communities: Optional[Dict[str, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: Union[int, float] = 1.0,
    randomness: Union[int, float] = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: Union[int, float] = 1.0,
    check_directed: bool = True,
) -> HierarchicalClusters:
    """

    Leiden is a global network partitioning algorithm. Given a graph, it will iterate
    through the network node by node, and test for an improvement in our quality
    maximization function by speculatively joining partitions of each neighboring node.

    This process continues until no moves are made that increases the partitioning
    quality.

    Unlike the function :func:`customleiden.partition.leiden`, this function does not
    stop after maximization has been achieved. On some large graphs, it's useful to
    identify particularly large communities whose membership counts exceed
    ``max_cluster_size`` and induce a subnetwork solely out of that community. This
    subnetwork is then treated as a wholly separate entity, leiden is run over it, and
    the new, smaller communities are then mapped into the original community map space.

    The results also differ substantially; the returned List[HierarchicalCluster] is
    more of a log of state at each level. All HierarchicalClusters at level 0 should be
    considered to be the results of running :func:`customleiden.partition.leiden`. Every
    community whose membership is greater than ``max_cluster_size`` will then
    also have entries where level == 1, and so on until no communities are greater in
    population than ``max_cluster_size`` OR we are unable to break them down any
    further.

    Once a node's membership registration in a community cannot be changed any further,
    it is marked with the flag
    ``customleiden.partition.HierarchicalCluster.is_final_cluster = True``.

    Parameters
    ----------
    graph : Union[List[Tuple[Any, Any, Union[int, float]]], GraphRepresentation]
        A graph representation, whether a weighted edge list referencing an undirected
        graph, an undirected networkx graph, or an undirected adjacency matrix in either
        numpy.ndarray or scipy.sparse.csr_array form. Please see the Notes section
        regarding node ids used.
    max_cluster_size : int
        Default is ``1000``. Any partition or cluster with
        membership >= ``max_cluster_size`` will be isolated into a subnetwork. This
        subnetwork will be used for a new leiden global partition mapping, which will
        then be remapped back into the global space after completion. Once all
        clusters with membership >= ``max_cluster_size`` have been completed, the level
        increases and the partition scheme is scanned again for any new clusters with
        membership >= ``max_cluster_size`` and the process continues until every
        cluster's membership is < ``max_cluster_size`` or if they cannot be broken into
        more than one new community.
    starting_communities : Optional[Dict[Any, int]]
        Default is ``None``. An optional community mapping dictionary that contains a node
        id mapping to the community it belongs to. Please see the Notes section regarding
        node ids used.

        If no community map is provided, the default behavior is to create a node
        community identity map, where every node is in their own community.
    extra_forced_iterations : int
        Default is ``0``. Leiden will run until a maximum quality score has been found
        for the node clustering and no nodes are moved to a new cluster in another
        iteration. As there is an element of randomness to the Leiden algorithm, it is
        sometimes useful to set ``extra_forced_iterations`` to a number larger than 0
        where the entire process is forced to attempt further refinement.
    resolution : Union[int, float]
        Default is ``1.0``. Higher resolution values lead to more communities and lower
        resolution values leads to fewer communities. Must be greater than 0.
    randomness : Union[int, float]
        Default is ``0.001``. The larger the randomness value, the more exploration of
        the partition space is possible. This is a major difference from the Louvain
        algorithm, which is purely greedy in the partition exploration.
    use_modularity : bool
        Default is ``True``. If ``False``, will use a Constant Potts Model (CPM).
    random_seed : Optional[int]
        Default is ``None``. Can provide an optional seed to the PRNG used in Leiden
        for deterministic output.
    weight_attribute : str
        Default is ``weight``. Only used when creating a weighed edge list of tuples
        when the source graph is a networkx graph. This attribute corresponds to the
        edge data dict key.
    is_weighted : Optional[bool]
        Default is ``None``. Only used when creating a weighted edge list of tuples
        when the source graph is an adjacency matrix. The
        :func:`customleiden.utils.is_unweighted` function will scan these
        matrices and attempt to determine whether it is weighted or not. This flag can
        short circuit this test and the values in the adjacency matrix will be treated
        as weights.
    weight_default : Union[int, float]
        Default is ``1.0``. If the graph is a networkx graph and the graph does not
        have a fully weighted sequence of edges, this default will be used. If the
        adjacency matrix is found or specified to be unweighted, this weight_default
        will be used for every edge.
    check_directed : bool
        Default is ``True``. If the graph is an adjacency matrix, we will attempt to
        ascertain whether it is directed or undirected. As our leiden implementation is
        only known to work with an undirected graph, this function will raise an error
        if it is found to be a directed graph. If you know it is undirected and wish to
        avoid this scan, you can set this value to ``False`` and only the lower triangle
        of the adjacency matrix will be used to generate the weighted edge list.

    Returns
    -------
    HierarchicalClusters
        The results of running hierarchical leiden over the provided graph, a list of
        HierarchicalClusters identifying the state of every node and cluster at each
        level. Isolate nodes in the input graph are not returned in the result.

    Raises
    ------
    ValueError
    TypeError
    BeartypeCallHintParamViolation

    See Also
    --------
    customleiden.utils.is_unweighted

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
        guaranteeing well-connected communities",Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/graspologic-org/graspologic-native

    Notes
    -----
    No two different nodes are allowed to encode to the **same** str representation,
    e.g. node_a id of ``"1"`` and node_b id of ``1`` are different object types
    but str(node_a) == str(node_b). This collision will result in a ``ValueError``

    This function is implemented in the `graspologic-native` Python module, a module
    written in Rust for Python.
    """
    _validate_common_arguments(
        extra_forced_iterations,
        resolution,
        randomness,
        random_seed,
    )
    check_argument(max_cluster_size > 0, "max_cluster_size must be a positive int")

    identifier = _IdentityMapper()
    node_count: int
    edges: List[Tuple[str, str, float]]
    if isinstance(graph, nx.Graph):
        node_count, edges = _nx_to_edge_list(
            graph, identifier, is_weighted, weight_attribute, weight_default
        )
    elif isinstance(graph, list):
        node_count, edges = _edge_list_to_edge_list(graph, identifier)
    else:
        node_count, edges = _adjacency_matrix_to_edge_list(
            graph, identifier, check_directed, is_weighted, weight_default
        )

    native_friendly_communities = _community_python_to_native(
        starting_communities, identifier
    )

    hierarchical_clusters_native = gn.hierarchical_leiden(
        edges=edges,
        starting_communities=native_friendly_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        max_cluster_size=max_cluster_size,
        seed=random_seed,
    )

    result_partitions = HierarchicalClusters()
    all_nodes = set()
    for entry in hierarchical_clusters_native:
        partition = _from_native(entry, identifier)
        result_partitions.append(partition)
        all_nodes.add(partition.node)

    if len(all_nodes) < node_count:
        warnings.warn(
            "Leiden partitions do not contain all nodes from the input graph because input graph "
            "contained isolate nodes."
        )

    return result_partitions

class LeidenContextResult(NamedTuple):
    partitions: Dict[Any, int]
    context_nodes: Dict[int, Set[Any]]
    cluster_graph: nx.Graph

@beartype
def compute_node2vec_embeddings(
    graph: nx.Graph, dimensions: int = 64
) -> Dict[str, np.ndarray]:
    """
    Computes dense vector embeddings for each node in the graph using the node2vec algorithm.

    node2vec is a biased random walk–based embedding method that captures both homophily 
    (similar nodes) and structural equivalence (similar roles).

    Parameters
    ----------
    graph : nx.Graph
        An undirected NetworkX graph whose nodes will be embedded.
    dimensions : int
        Dimensionality of the embedding space. Defaults to 64.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary mapping node ID (as string) to its vector embedding.
        These embeddings can be used for downstream similarity-based analysis.
    
    Notes
    -----
    node2vec simulates biased random walks over the graph and trains a 
    Word2Vec model on the resulting walk sequences.
    """
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=200, workers=1)
    model = node2vec.fit(window=10, min_count=1)
    return {str(node): model.wv[str(node)] for node in graph.nodes()}

def _compute_embeddings(
    graph: nx.Graph,
    method: Union[str, Callable[[nx.Graph], Dict[str, np.ndarray]]] = "node2vec",
    *,
    dimensions: int = 64,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, np.ndarray]:
    """
    Unified interface for computing node embeddings using different strategies.

    Supports:
    - "node2vec": structure-based embeddings
    - "sbert" (via SentenceTransformer): content-based embeddings from node text

    Parameters
    ----------
    graph : nx.Graph
        Input graph to embed. If using "sbert", it is assumed each node has a "text" attribute.
    method : Union[str, Callable]
        Embedding method: "node2vec", "sbert", or a custom callable that returns node embeddings.
    dimensions : int
        Embedding size for node2vec. Ignored for sbert.
    model_name : str
        SentenceTransformer model name (e.g., "all-MiniLM-L6-v2"). Only used if method == "sbert".

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping node ID (as str) to vector embeddings.

    Raises
    ------
    ValueError
        If an unknown method string is provided.
    """
    if callable(method):
        return method(graph)

    if method == "node2vec":
        n2v = Node2Vec(graph, dimensions=dimensions,
                       walk_length=30, num_walks=200, workers=1)
        wv = n2v.fit(window=10, min_count=1).wv
        return {k: wv[k] for k in wv.key_to_index}

    if method == "sbert":
        mdl = SentenceTransformer(model_name)
        # assumes node text stored in node attr "text"
        texts = [graph.nodes[n].get("text", str(n)) for n in graph.nodes]
        embs = mdl.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return {str(n): e for n, e in zip(graph.nodes, embs)}

    raise ValueError(f"Unknown embedding method: {method}")

def detect_misalignment(
    partitions: Dict[Any, int],
    embeddings: Dict[str, np.ndarray],
) -> Dict[Any, Tuple[int, float]]:
    """
    Detects semantic misalignment between assigned clusters and node similarity in embedding space.

    For each node, this function checks whether its average similarity to another community
    is higher than to its own assigned community. If so, the node is considered "misaligned".

    Parameters
    ----------
    partitions : Dict[Any, int]
        Mapping of node ID to its assigned cluster.
    embeddings : Dict[str, np.ndarray]
        Mapping of node ID (as str) to its embedding vector.

    Returns
    -------
    Dict[Any, Tuple[int, float]]
        Misaligned nodes mapped to:
        - their most similar alternative community
        - the corresponding average similarity score

    Notes
    -----
    Misalignment is a useful signal for selecting context nodes or diagnosing
    over-merged or weakly connected clusters.
    """
    community_to_nodes = defaultdict(list)
    for n, c in partitions.items():
        community_to_nodes[c].append(n)

    mis = {}
    for n, c in partitions.items():
        n_emb = embeddings[str(n)]
        best_c, best_sim = c, -1.0
        for oc, members in community_to_nodes.items():
            if oc == c:
                continue
            sims = [cosine_similarity([n_emb], [embeddings[str(m)]])[0, 0]
                    for m in members if str(m) in embeddings]
            if sims:
                avg = float(np.mean(sims))
                if avg > best_sim:
                    best_sim, best_c = avg, oc
        if best_c != c:
            mis[n] = (best_c, best_sim)
    return mis

def _infer_defaults(
    graph: nx.Graph,
    lambda_max: int,
    alpha: float,
    beta: float,
    partitions: dict[Any, int],
) -> tuple[int, float, float]:
    """
    Automatically infers default values for lambda_max, alpha, and beta based on graph structure.

    These parameters control the selection of context nodes:
    - lambda_max : maximum number of context nodes per community
    - alpha : controls the adaptive size of lambda_c based on log(#candidates)
    - beta : penalty for redundancy when selecting context nodes (MMR-style)

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    lambda_max : Optional[int]
        User-specified upper bound for lambda_c. If None, will be inferred from graph size.
    alpha : Optional[float]
        Scaling factor for adaptive lambda_c. If None, set to 1 + graph density.
    beta : Optional[float]
        Redundancy penalty for context selection. If None, inferred from modularity.

    partitions : dict[Any, int]
        Community assignments from Leiden. Used to estimate modularity for beta.

    Returns
    -------
    tuple[int, float, float]
        Finalized values for (lambda_max, alpha, beta)
    """
    if lambda_max is None:
        lambda_max = max(2, math.ceil(math.log2(graph.number_of_nodes())))

    if alpha is None:
        dens = nx.density(graph)                       # 0 … 1
        alpha = 1.0 + dens                             # 1 … 2

    if beta is None:
        try:
            from networkx.algorithms.community.quality import modularity
            comms = {}
            for n, c in partitions.items():
                comms.setdefault(c, []).append(n)
            mod = modularity(graph, comms.values())    # –0.5 … 1
        except Exception:                              # fallback
            mod = 0.2
        beta = min(0.7, max(0.3, 0.4 + mod))           # clamp to [0.3,0.7]

    return lambda_max, alpha, beta

def _build_context(
    graph: nx.Graph,
    partitions: Dict[Any, int],
    embeddings: Dict[str, np.ndarray],
    misaligned: Dict[Any, Tuple[int, float]],
    *,
    lambda_max: int,
    alpha: float,
    beta: float,
    use_betweenness_penalty: bool = False,
) -> Tuple[Dict[int, Set[Any]], nx.Graph]:
    """
    Selects context nodes per community based on semantic misalignment and graph topology.

    This internal function identifies a small set of representative "context nodes" for 
    each community based on how semantically similar they are to nodes in other communities 
    and how topologically close (in hop distance) they are to neighbors in different clusters.
    
    The function uses a hybrid scoring function and greedy MMR-style selection to balance 
    relevance and diversity. The result is useful for summarization, interpretability, and 
    building coarse cluster-level graphs.

    Parameters
    ----------
    graph : nx.Graph
        Input undirected graph. Must contain edge weights if relevant. Nodes can optionally 
        have a "text" attribute if using SBERT embeddings.

    partitions : Dict[Any, int]
        Mapping from node ID to cluster/community ID. Usually produced by the Leiden algorithm.

    embeddings : Dict[str, np.ndarray]
        Embedding vectors for each node. Keys must be `str(node_id)` and values are 
        vector embeddings (e.g., node2vec, SBERT).

    misaligned : Dict[Any, Tuple[int, float]]
        Misaligned nodes with their suggested alternative cluster and similarity score.
        Typically computed using `detect_misalignment(...)`.

    lambda_max : int
        Maximum number of context nodes to select per community.

    alpha : float
        Controls the number of selected context nodes per community based on:
        `lambda_c = ceil(alpha * log2(candidate_pool + 1))`, clamped by `lambda_max`.

    beta : float
        Diversity penalty. If > 0, selected context nodes are chosen using a 
        Maximum Marginal Relevance (MMR)-style formula to reduce redundancy.

    use_betweenness_penalty : bool, default=False
        If True, penalizes edges with high edge betweenness centrality during hop 
        length computation. This helps avoid selecting overly-central nodes.

    Returns
    -------
    Tuple[Dict[int, Set[Any]], nx.Graph]
        - context_nodes : Dict[cluster_id, Set[node]]
            Key context nodes selected for each community.
        - cluster_graph : nx.Graph
            Coarse cluster-level graph where nodes are communities and edges represent
            inter-cluster interactions aggregated from the original graph.

    Notes
    -----
    A node is a strong candidate for context selection if:
    - It has a high semantic similarity to nodes in a different cluster
    - It lies on short paths (small hop count) to external neighbors
    - It helps represent community boundaries or bridge information between clusters

    Each community’s candidate context nodes are scored using:
        score = similarity / (hop + 1)

    Final context selection is done greedily with MMR-style penalty:
        score_i - beta * max(similarity_to_selected)
    """

    # Edge centrality for hop penalty
    edge_betweenness = (
        nx.edge_betweenness_centrality(graph, normalized=True)
        if use_betweenness_penalty else {}
    )

    # Annotate nodes with cluster label for clarity
    nx.set_node_attributes(graph, partitions, "cluster")

    inter_edges = []
    candidate_scores = defaultdict(list)
    node_to_hop: Dict[Any, int] = {}

    for u, v, data in graph.edges(data=True):
        cu, cv = partitions[u], partitions[v]
        if cu == cv:
            continue  # skip intra-cluster edges

        weight = float(data.get("weight", 1.0))
        inter_edges.append((cu, cv, weight))

        for node, src_comm, tgt_comm, neighbor in [(u, cu, cv, v), (v, cv, cu, u)]:
            if node not in misaligned:
                continue
            _, sim = misaligned[node]

            try:
                hop = nx.shortest_path_length(graph, source=node, target=neighbor)
            except nx.NetworkXNoPath:
                hop = 1

            if use_betweenness_penalty:
                edge_key = (min(node, neighbor), max(node, neighbor))
                hop *= 1 + edge_betweenness.get(edge_key, 0.0)

            score = sim / (hop + 1)
            candidate_scores[src_comm].append((node, score))
            node_to_hop[node] = int(round(hop))

    context_nodes: Dict[int, Set[Any]] = {}

    for comm, candidates in candidate_scores.items():
        lambda_c = min(lambda_max, max(1, math.ceil(alpha * math.log2(len(candidates) + 1))))
        selected: List[Any] = []

        candidates.sort(key=lambda x: -x[1])  # sort by score descending

        while len(selected) < lambda_c and candidates:
            best_idx, best_score = -1, -float("inf")
            for i, (node, base_score) in enumerate(candidates):
                if not selected or beta == 0:
                    mmr_score = base_score
                else:
                    sim_to_sel = max(
                        cosine_similarity(
                            [embeddings[str(node)]],
                            [embeddings[str(sel)]]  # each selected node
                        )[0, 0]
                        for sel in selected
                    )
                    mmr_score = base_score - beta * sim_to_sel

                if mmr_score > best_score:
                    best_idx, best_score = i, mmr_score

            node, _ = candidates.pop(best_idx)
            selected.append(node)

        context_nodes[comm] = set(selected)

        print(f"[CTX] Community {comm:<3} | lambda={lambda_c} | selected={len(selected)}")
        for node in selected:
            tgt_comm, sim = misaligned[node]
            hop_len = node_to_hop.get(node, "?")
            score_disp = f"{sim / (hop_len + 1):.3f}" if isinstance(hop_len, int) else "?"
            print(f"   • Node {node} → {tgt_comm} | sim={sim:.3f} | hop={hop_len} | score={score_disp}")

    # Cluster-level summary graph
    cluster_graph = nx.Graph()
    cluster_graph.add_weighted_edges_from(inter_edges)

    return context_nodes, cluster_graph

def leiden_with_context(
    graph: nx.Graph,
    *,
    embedding_method: Union[str, Callable] = "node2vec",
    lambda_max: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    use_betweenness_penalty: bool = False,
    **leiden_kwargs,
) -> LeidenContextResult:
    
    """
    Run the Leiden community detection algorithm on a graph and select representative 
    context nodes for each community using semantic similarity and hop distance.

    This function extends the standard Leiden clustering by identifying a small number 
    of informative "context nodes" for each detected community. These nodes are selected 
    based on their semantic misalignment (using embedding similarity) and their topological 
    position (hop distance from external neighbors), with optional MMR-style diversity control.

    The returned context nodes can be used for:
    - Constructing coarse-grained "cluster graphs"
    - Visual explanation or summarization of communities
    - Downstream reasoning in graph-based retrieval or analysis

    Parameters
    ----------
    graph : nx.Graph
        The input undirected graph, with nodes optionally containing a "text" attribute 
        (used by SBERT embeddings). Must not be a multigraph or directed.

    embedding_method : str or Callable, default="node2vec"
        Embedding method used to compute semantic similarity. If a string, must be:
        - "node2vec" : Learns embeddings from graph walks.
        - "sbert" : Uses Sentence-BERT on node "text" attributes.
        If a callable, it should accept the graph and return a dict of node → embedding.

    lambda_max : int, optional
        Maximum number of context nodes per community. If not provided, defaults to 
        `ceil(log2(N))` where N is the number of nodes in the graph.

    alpha : float, optional
        Controls how many context nodes are selected relative to the size of each 
        candidate pool. If not provided, inferred from graph density.

    beta : float, optional
        Controls the diversity penalty when selecting context nodes. Higher values 
        encourage diversity (MMR-style). Automatically inferred from modularity 
        if not specified.

    use_betweenness_penalty : bool, default=False
        If True, penalizes nodes with high edge betweenness during hop computation, 
        to reduce over-selection of topologically central nodes.

    **leiden_kwargs : additional keyword args
        Additional arguments passed to the base `leiden(...)` function, such as:
        - `random_seed`
        - `resolution`
        - `extra_forced_iterations`
        - `use_modularity`

    Returns
    -------
    LeidenContextResult
        A named tuple containing:
        - `partitions`: Dict[node, cluster_id] — final community assignments.
        - `context_nodes`: Dict[cluster_id, Set[node]] — key context nodes per community.
        - `cluster_graph`: A coarse graph where each node is a cluster and edges represent inter-cluster connections.

    Raises
    ------
    ValueError
        If input graph is directed or multigraph.
    NotImplementedError
        If embedding method is not recognized.
    BeartypeCallHintParamViolation
        If arguments violate runtime type checks.

    See Also
    --------
    customleiden.partition.leiden
    hierarchical_leiden_with_context
    compute_node2vec_embeddings
    detect_misalignment

    Notes
    -----
    This function uses semantic embeddings to detect "misaligned" nodes — nodes that 
    are more similar to a different community than the one assigned by the Leiden algorithm. 
    Such nodes are used as candidates for context selection, ensuring they are informative 
    bridges between clusters.

    A coarse cluster-level graph is constructed by collapsing the original graph into 
    communities, where edges represent inter-community connections.

    Context selection uses:
    - A hybrid scoring function: `similarity / (hop + 1)`
    - MMR (Maximum Marginal Relevance) to encourage diversity
    - Adaptive lambda per community based on candidate pool size
    """

    # base partitions
    partitions = leiden(graph, **leiden_kwargs)

    # concrete defaults
    lambda_max, alpha, beta = _infer_defaults(
        graph, lambda_max, alpha, beta, partitions
    )
    print(f"[INFO] lambda_max={lambda_max} | alpha={alpha:.2f} | beta={beta:.2f}")

    # embeddings & misalignment
    embeddings = _compute_embeddings(graph, embedding_method)
    misaligned  = detect_misalignment(partitions, embeddings)

    # context picking
    context, c_graph = _build_context(
        graph,
        partitions,
        embeddings,
        misaligned,
        lambda_max=lambda_max,
        alpha=alpha,
        beta=beta,
        use_betweenness_penalty=use_betweenness_penalty,
    )
    return LeidenContextResult(partitions, context, c_graph)

def hierarchical_leiden_with_context(
    graph: nx.Graph,
    *,
    max_cluster_size: int = 1000,
    embedding_method: Union[str, Callable] = "node2vec",
    lambda_max: Optional[int] = None,
    alpha: float,
    beta: float,
    use_betweenness_penalty: bool = False,
    **leiden_kwargs,
) -> List[LeidenContextResult]:
    """
    Runs hierarchical Leiden clustering and extracts semantically meaningful context nodes 
    for each community at each hierarchy level using similarity and structural signals.

    This function performs community detection using the hierarchical variant of the Leiden 
    algorithm. For each level in the hierarchy, it identifies a set of context nodes that 
    bridge across community boundaries. These context nodes are selected using a hybrid 
    score based on node embeddings, hop distance to external neighbors, and a diversity 
    penalty (MMR-style).

    The function returns rich metadata per level, including:
    - The partitions (community assignments)
    - The selected context nodes
    - A coarse cluster-level graph connecting communities

    Parameters
    ----------
    graph : nx.Graph
        Input undirected graph. Can contain weights or text attributes per node.
        Used as input to the Leiden clustering and similarity scoring.

    max_cluster_size : int, default=1000
        Threshold for recursive splitting in hierarchical Leiden. Communities with 
        membership greater than this size are broken into smaller communities in 
        deeper levels.

    embedding_method : Union[str, Callable], default="node2vec"
        Node embedding method to use for computing semantic similarity.
        - "node2vec" : Uses random-walk-based embeddings
        - "sbert" : Uses Sentence-BERT on node "text" attribute
        - Callable : Custom embedding function with signature `fn(graph) -> Dict[str, np.ndarray]`

    lambda_max : int, optional
        Maximum number of context nodes per community. If None, inferred from graph size.

    alpha : float, default=1.0
        Controls adaptive selection of context nodes using:
        `lambda_c = ceil(alpha * log2(#candidates + 1))`
        Higher alpha → more context nodes (up to lambda_max)

    beta : float, default=0.5
        Controls diversity penalty in greedy selection (MMR).
        - 0.0: Pure relevance-based selection
        - >0.0: Penalizes redundant/overlapping context nodes

    use_betweenness_penalty : bool, default=False
        If True, adjusts hop-distance with edge betweenness penalty to avoid 
        selecting overly-central nodes.

    **leiden_kwargs : dict
        Additional keyword arguments passed to `hierarchical_leiden(...)`, such as:
        - resolution
        - randomness
        - random_seed
        - starting_communities
        - weight_attribute

    Returns
    -------
    List[LeidenContextResult]
        A list of results, one for each hierarchy level.
        Each result includes:
        - `.partitions`: node → cluster assignment
        - `.context_nodes`: selected context nodes per cluster
        - `.cluster_graph`: coarse graph of inter-cluster edges

    Notes
    -----
    This function is useful in:
    - Multi-resolution clustering (e.g., zoomable topic maps)
    - Context-aware summarization or QA
    - High-level structure over large, dense graphs
    """
    h_clusters = hierarchical_leiden(
        graph,
        max_cluster_size=max_cluster_size,
        **leiden_kwargs,
    )
    levels = max(h.level for h in h_clusters)
    results: List[LeidenContextResult] = []

    # Precompute embeddings once for all levels
    embeddings = _compute_embeddings(graph, embedding_method)

    for level in range(levels + 1):
        level_partitions = {
            h.node: h.cluster
            for h in h_clusters
            if h.level == level
        }

        misaligned = detect_misalignment(level_partitions, embeddings)

        # Use edges from graph directly
        context, cluster_graph = _build_context(
            graph,
            level_partitions,
            embeddings,
            misaligned,
            lambda_max=lambda_max,
            alpha=alpha,
            beta=beta,
            use_betweenness_penalty=use_betweenness_penalty,
        )

        results.append(
            LeidenContextResult(level_partitions, context, cluster_graph)
        )

    return results
