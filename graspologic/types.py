# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""
This module includes common graspologic type hint declarations.
"""

from typing import Any, List, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse.csr import csr_matrix


BasicUndirectedNetworkxGraph = Union[nx.Graph, nx.OrderedGraph]
BasicDirectedNetworkxGraph = Union[nx.DiGraph, nx.OrderedDiGraph]
BasicNetworkxGraph = Union[BasicUndirectedNetworkxGraph, BasicDirectedNetworkxGraph]
MultiNetworkxGraph = Union[nx.MultiGraph, nx.MultiDiGraph, nx.OrderedMultiGraph, nx.OrderedMultiDiGraph]
NetworkxGraph = Union[BasicNetworkxGraph, MultiNetworkxGraph]

DenseAdjacencyMatrix = Union[np.ndarray, np.memmap]
SparseAdjacencyMatrix = csr_matrix
AdjacencyMatrix = Union[DenseAdjacencyMatrix, SparseAdjacencyMatrix]

GraphRepresentation = Union[BasicNetworkxGraph, AdjacencyMatrix]

AnyEdges = List[Tuple[Any, Any, Union[float, int]]]
StringEdges = List[Tuple[str, str, Union[float, int]]]
IntEdges = List[Tuple[int, int, Union[float, int]]]


__all__ = [
    "AdjacencyMatrix",
    "BasicNetworkxGraph",
    "BasicDirectedNetworkxGraph",
    "BasicUndirectedNetworkxGraph",
    "DenseAdjacencyMatrix",
    "GraphRepresentation",
    "MultiNetworkxGraph",
    "NetworkxGraph",
    "SparseAdjacencyMatrix"
]