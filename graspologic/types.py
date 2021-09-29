# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""
This module includes common graspologic type hint declarations.
"""

from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse as sp

AdjacencyMatrix = Union[np.ndarray, sp.csr_matrix]

GraphRepresentation = Union[np.ndarray, sp.csr_matrix, nx.Graph]

__all__ = ["AdjacencyMatrix", "GraphRepresentation"]
