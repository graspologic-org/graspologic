# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Type, Union

import networkx as nx


def assert_nx_graph(
    graph: Any,
    message: str = f"Graph provided must be of type networkx.Graph. Note that networkx.Graph has a number of subclasses such as DiGraph, or MultiGraph",
):
    if not isinstance(graph, nx.Graph):
        raise TypeError(message)


def assert_simple_nx_graph(
    graph: Any,
    message: str = "Graph provided must be of type networkx.Graph or networkx.DiGraph.",
):
    assert_nx_graph(graph)
    graph: nx.Graph = graph
    if graph.is_multigraph():
        raise TypeError(message)


# common assertions
def assert_weighted(
    graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
    message: str = "The graph provided must be fully weighted.",
    weight_attribute: str = "weight",
    error_type: Type = ValueError,
):
    if not nx.is_weighted(graph, weight=weight_attribute):
        raise error_type(message)
