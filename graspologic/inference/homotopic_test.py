# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
import networkx as nx

_VALID_TESTS = ["fisher", "mww"]

def homotypic_test(
    graph: Union[nx.Graph, nx.DiGraph],
    test: str,
) -> float:
    """
    Returns test of homotypic affinity in a graph.
    """
    if not isinstance(graph, Union[nx.Graph, nx.DiGraph]):
        msg = "Graph must be a nxGraph, not {}".format(type(graph))
        raise TypeError(msg)

    if not isinstance(test, str):
        msg = "Test must be a str, not {}".format(type(test))
        raise TypeError(msg)
    elif test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
        raise ValueError(msg)

    graph_array = nx.to_numpy_array(graph)

    m, m = graph_array.shape
    comm_size = int(0.5*m)
    comm_edges = comm_size**2

    if test == "fisher":
        con_table = np.zeros((2, 2))

        c_1_only = graph[:half_point, :half_point]
        c_2_only = graph[half_point:, half_point:]
        c_1_c_2 = graph[:half_point, half_point:]
        c_2_c_1 = graph[half_point:, :half_point]

        c_1_edge_num = find_edge(c_1_only)
        c_2_edge_num = find_edge(c_2_only)

        c_1_edge_num = c_1_upper_edge_num + c_1_lower_edge_num


        c_2_upper = graph[:half_point, half_point:]
        c_2_lower = graph[half_point:, :half_point]

        c_2_upper_edge_num = len(c_1_upper[c_1_upper != 0])
        c_2_lower_edge_num = len(c_1_lower[c_1_lower != 0])

        c_2_edge_num = c_2_upper_edge_num + c_2_lower_edge_num

        con_table[0, 0] = c_2_edge_num / comm_edges

def find_edge(arr):

    return arr[arr != 0]