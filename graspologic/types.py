# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""
This module includes common graspologic type hint declarations.
"""

import sys
from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse as sp

if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import list as List
    from builtins import set as Set
    from builtins import tuple as Tuple
else:
    from typing import Dict, List, Set, Tuple

AdjacencyMatrix = Union[np.ndarray, sp.csr_matrix]

GraphRepresentation = Union[np.ndarray, sp.csr_matrix, nx.Graph]

__all__ = ["AdjacencyMatrix", "Dict", "List", "GraphRepresentation", "Set", "Tuple"]
