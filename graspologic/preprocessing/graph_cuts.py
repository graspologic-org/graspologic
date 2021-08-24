# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import logging
import random
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import networkx as nx
import numpy as np

LARGER_THAN_INCLUSIVE = "larger_than_inclusive"
LARGER_THAN_EXCLUSIVE = "larger_than_exclusive"
SMALLER_THAN_INCLUSIVE = "smaller_than_inclusive"
SMALLER_THAN_EXCLUSIVE = "smaller_than_exclusive"


def _filter_function_for_make_cuts(
    cut_threshold: Union[int, float],
    cut_process: str,
) -> Callable[[Tuple[Any, Union[int, float]]], bool]:
    filter_functions = {
        LARGER_THAN_EXCLUSIVE: lambda x: x[1] > cut_threshold,
        LARGER_THAN_INCLUSIVE: lambda x: x[1] >= cut_threshold,
        SMALLER_THAN_EXCLUSIVE: lambda x: x[1] < cut_threshold,
        SMALLER_THAN_INCLUSIVE: lambda x: x[1] <= cut_threshold,
    }
    if cut_process not in filter_functions:
        raise ValueError(f"Provided cut_process '{cut_process}' is not a valid value")
    return filter_functions[cut_process]


class DefinedHistogram(NamedTuple):
    """
    Contains the histogram and the edges of the bins in the histogram.
    The bin_edges will have a length 1 greater than the histogram, as it defines the
    minimal and maximal edges as well as each edge in between.
    """

    histogram: np.ndarray
    bin_edges: np.ndarray


def histogram_edge_weight(
    graph: Union[nx.Graph, nx.DiGraph],
    bin_directive: Union[int, List[Union[float, int]], np.ndarray, str] = 10,
    weight_attribute: str = "weight",
) -> DefinedHistogram:
    """
    Generates a histogram of the edge weights of the provided graph. Histogram function
    is fundamentally proxied through to numpy's `histogram` function, and bin selection
    follows :func:`numpy.histogram` processes.

    Parameters
    ----------

    graph : nx.Graph
        The graph. No changes will be made to it.
    bin_directive : Union[int, List[Union[float, int]], numpy.ndarray, str]
        Is passed directly through to numpy's "histogram" (and thus,
        "histogram_bin_edges") functions.

        See: :func:`numpy.histogram_bin_edges`

        In short: if an int is provided, we use ``bin_directive`` number of equal range
        bins.

        If a sequence is provided, these bin edges will be used and can be sized to
        whatever size you prefer

        Note that the :class:`numpy.ndarray` should be ndim=1 and the values should be
        float or int.
    weight_attribute : str
        The weight attribute name in the data dictionary. Default is `weight`.

    Returns
    -------
    :class:`DefinedHistogram`
        A named tuple that contains the histogram and the bin_edges used in the
        histogram

    Notes
    -----
    Edges without a `weight_attribute` field will be excluded from this histogram.
    Enable logging to view any messages about edges without weights.
    """
    logger = logging.getLogger(__name__)
    edge_weights: List[Union[int, float, None]] = [
        weight for _, _, weight in graph.edges(data=weight_attribute)
    ]
    none_weights: List[None] = [weight for weight in edge_weights if weight is None]
    actual_weights: List[Union[int, float]] = [
        weight for weight in edge_weights if weight is not None
    ]

    if len(none_weights) != 0:
        logger.warning(
            f"Graph contains {len(none_weights)} edges with no {weight_attribute}."
            + f" Histogram excludes these values."
        )

    histogram, bin_edges = np.histogram(actual_weights, bin_directive)

    return DefinedHistogram(histogram=histogram, bin_edges=bin_edges)


def cut_edges_by_weight(
    graph: Union[nx.Graph, nx.DiGraph],
    cut_threshold: Union[int, float],
    cut_process: str,
    weight_attribute: str = "weight",
    prune_isolates: bool = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Thresholds edges (removing them from the graph and returning a copy) by weight.

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        The graph that will be copied and pruned.
    cut_threshold : Union[int, float]
        The threshold for making cuts based on weight.
    cut_process : str
        Describes how we should make the cut; cut all edges larger or smaller than the
        cut_threshold, and whether exclusive or inclusive. Allowed values are

        - ``larger_than_inclusive``
        - ``larger_than_exclusive``
        - ``smaller_than_inclusive``
        - ``smaller_than_exclusive``
    weight_attribute : str
        The weight attribute name in the edge's data dictionary. Default is `weight`.
    prune_isolates : bool
        If true, remove any vertex that no longer has an edge.  Note that this only
        prunes vertices which have edges to be pruned; any isolate vertex prior to any
        edge cut will be retained.

    Returns
    -------
    Union[nx.Graph, nx.DiGraph]
        Pruned copy of the same type of graph provided

    Notes
    -----
    Edges without a `weight_attribute` field will be excluded from these cuts.  Enable
    logging to view any messages about edges without weights.
    """
    filter_by = _filter_function_for_make_cuts(cut_threshold, cut_process)
    if not isinstance(cut_threshold, int) and not isinstance(cut_threshold, float):
        raise TypeError(
            f"cut_threshold must be of type int or float; you provided: "
            f"{type(cut_threshold)}"
        )

    logger = logging.getLogger(__name__)
    graph_copy = graph.copy()
    edge_weights: List[Tuple[Tuple[Any, Any], Union[int, float, None]]] = [
        ((source, target), weight)
        for source, target, weight in graph.edges(data=weight_attribute)
    ]
    none_weights: List[Tuple[Tuple[Any, Any], None]] = [
        (edge, weight) for edge, weight in edge_weights if weight is None
    ]
    actual_weights: List[Tuple[Tuple[Any, Any], Union[int, float]]] = [
        (edge, weight) for edge, weight in edge_weights if weight is not None
    ]

    if len(none_weights) != 0:
        logger.warning(
            f"Graph contains {len(none_weights)} edges with no {weight_attribute}."
            + f"Ignoring these when cutting by weight"
        )

    edges_to_cut = [x for x in actual_weights if filter_by(x)]
    for edge, weight in edges_to_cut:
        source, target = edge
        if (
            source in graph_copy
            and target in graph_copy
            and target in graph_copy[source]
        ):
            graph_copy.remove_edge(source, target)
        if prune_isolates:
            if len(graph_copy[source]) == 0:
                graph_copy.remove_node(source)
            if len(graph_copy[target]) == 0:
                graph_copy.remove_node(target)

    return graph_copy


def histogram_degree_centrality(
    graph: Union[nx.Graph, nx.DiGraph],
    bin_directive: Union[int, List[Union[float, int]], np.ndarray, str] = 10,
) -> DefinedHistogram:
    """
    Generates a histogram of the vertex degree centrality of the provided graph.
    Histogram function is fundamentally proxied through to numpy's `histogram` function,
    and bin selection follows :func:`numpy.histogram` processes.

    Parameters
    ----------

    graph : Union[nx.Graph, nx.DiGraph]
        The graph. No changes will be made to it.
    bin_directive : Union[int, List[Union[float, int]], numpy.ndarray, str]
        Is passed directly through to numpy's "histogram" (and thus,
        "histogram_bin_edges") functions.

        See: :func:`numpy.histogram_bin_edges`

        In short: if an int is provided, we use ``bin_directive`` number of equal range
        bins.

        If a sequence is provided, these bin edges will be used and can be sized to
        whatever size you prefer

        Note that the :class:`numpy.ndarray` should be ndim=1 and the values should be
        float or int.

    Returns
    -------
    :class:`DefinedHistogram`
        A named tuple that contains the histogram and the bin_edges used in the
        histogram
    """

    degree_centrality_dict = nx.degree_centrality(graph)
    histogram, bin_edges = np.histogram(
        list(degree_centrality_dict.values()), bin_directive
    )
    return DefinedHistogram(histogram=histogram, bin_edges=bin_edges)


def cut_vertices_by_degree_centrality(
    graph: Union[nx.Graph, nx.DiGraph],
    cut_threshold: Union[int, float],
    cut_process: str,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Given a graph and a cut_threshold and a cut_process, return a copy of the graph
    with the vertices outside of the cut_threshold.

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        The graph that will be copied and pruned.
    cut_threshold : Union[int, float]
        The threshold for making cuts based on weight.
    cut_process : str
        Describes how we should make the cut; cut all edges larger or smaller than the
        cut_threshold, and whether exclusive or inclusive. Allowed values are

        - ``larger_than_inclusive``
        - ``larger_than_exclusive``
        - ``smaller_than_inclusive``
        - ``smaller_than_exclusive``

    Returns
    -------
    Union[nx.Graph, nx.DiGraph]
        Pruned copy of the same type of graph provided
    """
    graph_copy = graph.copy()
    degree_centrality_dict = nx.degree_centrality(graph_copy)
    filter_by = _filter_function_for_make_cuts(cut_threshold, cut_process)
    vertices_to_cut = list(filter(filter_by, degree_centrality_dict.items()))
    for vertex, degree_centrality in vertices_to_cut:
        graph_copy.remove_node(vertex)

    return graph_copy


def histogram_betweenness_centrality(
    graph: Union[nx.Graph, nx.DiGraph],
    bin_directive: Union[int, List[Union[float, int]], np.ndarray, str] = 10,
    num_random_samples: Optional[int] = None,
    normalized: bool = True,
    weight_attribute: Optional[str] = "weight",
    include_endpoints: bool = False,
    random_seed: Optional[Union[int, random.Random, np.random.RandomState]] = None,
) -> DefinedHistogram:
    """
    Generates a histogram of the vertex betweenness centrality of the provided graph.
    Histogram function is fundamentally proxied through to numpy's `histogram` function,
    and bin selection follows :func:`numpy.histogram` processes.

    The betweenness centrality calculation can take advantage of networkx'
    implementation of randomized sampling by providing num_random_samples (or ``k``,
    in networkx betweenness_centrality nomenclature).

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        The graph. No changes will be made to it.
    bin_directive : Union[int, List[Union[float, int]], numpy.ndarray, str]
        Is passed directly through to numpy's "histogram" (and thus,
        "histogram_bin_edges") functions.

        See: :func:`numpy.histogram_bin_edges`

        In short: if an int is provided, we use ``bin_directive`` number of equal
        range bins.

        If a sequence is provided, these bin edges will be used and can be sized to
        whatever size you prefer

        Note that the :class:`numpy.ndarray` should be ndim=1 and the values should be
        float or int.
    num_random_samples : Optional[int]
        Use num_random_samples for vertex samples to *estimate* betweeness.
        num_random_samples should be <= len(graph.nodes). The larger num_random_samples
        is, the better the approximation. Default is ``None``.
    normalized : bool
        If True the betweenness values are normalized by :math:`2/((n-1)(n-2))` for
        undirected graphs, and  :math:`1/((n-1)(n-2))` for directed graphs where n is
        the number of vertices in the graph. Default is ``True``
    weight_attribute : Optional[str]
        If None, all edge weights are considered equal. Otherwise holds the name of the
        edge attribute used as weight. Default is ``weight``
    include_endpoints : bool
        If True include the endpoints in the shortest path counts.  Default is ``False``
    random_seed : Optional[Union[int, random.Random, np.random.RandomState]]
        Random seed or preconfigured random instance to be used for selecting random
        samples. Only used if num_random_samples is set. None will generate a new
        random state. Specifying a random state will provide consistent results between
        runs.

    Returns
    -------
    :class:`DefinedHistogram`
        A named tuple that contains the histogram and the bin_edges used in the
        histogram

    See Also
    --------
    networkx.algorithms.centrality.betweenness_centrality
    """

    betweenness_centrality_dict = nx.betweenness_centrality(
        G=graph,
        k=num_random_samples,
        normalized=normalized,
        weight=weight_attribute,
        endpoints=include_endpoints,
        seed=random_seed,
    )
    histogram, bin_edges = np.histogram(
        list(betweenness_centrality_dict.values()), bin_directive
    )
    return DefinedHistogram(histogram=histogram, bin_edges=bin_edges)


def cut_vertices_by_betweenness_centrality(
    graph: Union[nx.Graph, nx.DiGraph],
    cut_threshold: Union[int, float],
    cut_process: str,
    num_random_samples: Optional[int] = None,
    normalized: bool = True,
    weight_attribute: Optional[str] = "weight",
    include_endpoints: bool = False,
    random_seed: Optional[Union[int, random.Random, np.random.RandomState]] = None,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Given a graph and a cut_threshold and a cut_process, return a copy of the graph
    with the vertices outside of the cut_threshold.

    The betweenness centrality calculation can take advantage of networkx'
    implementation of randomized sampling by providing num_random_samples (or k, in
    networkx betweenness_centrality nomenclature).

    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        The graph that will be copied and pruned.
    cut_threshold : Union[int, float]
        The threshold for making cuts based on weight.
    cut_process : str
        Describes how we should make the cut; cut all edges larger or smaller than the
        cut_threshold, and whether exclusive or inclusive. Allowed values are

        - ``larger_than_inclusive``
        - ``larger_than_exclusive``
        - ``smaller_than_inclusive``
        - ``smaller_than_exclusive``
    num_random_samples : Optional[int]
        Use num_random_samples for vertex samples to *estimate* betweenness.
        num_random_samples should be <= len(graph.nodes). The larger num_random_samples
        is, the better the approximation. Default is ``None``.
    normalized : bool
        If True the betweenness values are normalized by :math:`2/((n-1)(n-2))` for
        undirected graphs, and :math:`1/((n-1)(n-2))` for directed graphs where n is
        the number of vertices in the graph. Default is ``True``
    weight_attribute : Optional[str]
        If None, all edge weights are considered equal. Otherwise holds the name of the
        edge attribute used as weight. Default is ``weight``
    include_endpoints : bool
        If True include the endpoints in the shortest path counts.  Default is ``False``
    random_seed : Optional[Union[int, random.Random, np.random.RandomState]]
        Random seed or preconfigured random instance to be used for selecting random
        samples. Only used if num_random_samples is set. None will generate a new
        random state. Specifying a random state will provide consistent results between
        runs.

    Returns
    -------
    Union[nx.Graph, nx.DiGraph]
        Pruned copy of the same type of graph provided

    See Also
    --------
    networkx.algorithms.centrality.betweenness_centrality
    """
    graph_copy = graph.copy()
    betweenness_centrality_dict = nx.betweenness_centrality(
        G=graph,
        k=num_random_samples,
        normalized=normalized,
        weight=weight_attribute,
        endpoints=include_endpoints,
        seed=random_seed,
    )
    filter_by = _filter_function_for_make_cuts(cut_threshold, cut_process)
    vertices_to_cut = list(filter(filter_by, betweenness_centrality_dict.items()))
    for vertex, degree_centrality in vertices_to_cut:
        graph_copy.remove_node(vertex)

    return graph_copy
