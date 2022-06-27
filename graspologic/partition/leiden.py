# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, NamedTuple, Optional, Union

import graspologic_native as gn
import networkx as nx
import numpy as np
import scipy
from beartype import beartype

from graspologic.types import AdjacencyMatrix, Dict, GraphRepresentation, List, Tuple

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
            "graphs of type np.ndarray or csr.sparse.csr.csr_matrix should be "
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
    for (node_id, partition) in starting_communities.items():
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
) -> Dict[Any, int]:
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
        numpy.ndarray or scipy.sparse.csr.csr_matrix form. Please see the Notes section
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
        :func:`graspologic.utils.is_unweighted` function will scan these
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
    graspologic.utils.is_unweighted

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
         guaranteeing well-connected communities", Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/microsoft/graspologic-native

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
    state log of our :func:`graspologic.partition.hierarchical_leiden` process
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
        scipy.sparse.csr.csr_matrix,
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

    Unlike the function :func:`graspologic.partition.leiden`, this function does not
    stop after maximization has been achieved. On some large graphs, it's useful to
    identify particularly large communities whose membership counts exceed
    ``max_cluster_size`` and induce a subnetwork solely out of that community. This
    subnetwork is then treated as a wholly separate entity, leiden is run over it, and
    the new, smaller communities are then mapped into the original community map space.

    The results also differ substantially; the returned List[HierarchicalCluster] is
    more of a log of state at each level. All HierarchicalClusters at level 0 should be
    considered to be the results of running :func:`graspologic.partition.leiden`. Every
    community whose membership is greater than ``max_cluster_size`` will then
    also have entries where level == 1, and so on until no communities are greater in
    population than ``max_cluster_size`` OR we are unable to break them down any
    further.

    Once a node's membership registration in a community cannot be changed any further,
    it is marked with the flag
    ``graspologic.partition.HierarchicalCluster.is_final_cluster = True``.

    Parameters
    ----------
    graph : Union[List[Tuple[Any, Any, Union[int, float]]], GraphRepresentation]
        A graph representation, whether a weighted edge list referencing an undirected
        graph, an undirected networkx graph, or an undirected adjacency matrix in either
        numpy.ndarray or scipy.sparse.csr.csr_matrix form. Please see the Notes section
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
        :func:`graspologic.utils.is_unweighted` function will scan these
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
    graspologic.utils.is_unweighted

    References
    ----------
    .. [1] Traag, V.A.; Waltman, L.; Van, Eck N.J. "From Louvain to Leiden:
        guaranteeing well-connected communities",Scientific Reports, Vol. 9, 2019
    .. [2] https://github.com/microsoft/graspologic-native

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
