# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""
This module includes common graspologic type hint declarations.
"""

import sys
from typing import Optional, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp

# an explanation, for those who come in the later times:
# the following is because when type hinting first came out, Python 3.6 up to 3.8
# (inclusive) specifically couldn't really handle having generics in the
# list/dict/tuple/set whatever primitives that are in builtins
# so we all used the typing module to provide their type signature
# and then 3.9 said 'oh, we can fix that now, and also if you use typing we'll cut you',
# so our choice was either to never support 3.9 onward, never support pre 3.9, or
# do this... jankery
# some things of note: mypy specifically looks for `sys.version_info` - so
# using a `from sys import version_info` gets ignored, and you will get mypy errors
# on top of that, doing `List = list` in the 3.9+ block doesn't work at all, so we
# have to use this VERY specific syntax.  if you want to test it, try it out, but as of
# today, `from builtins import foo as Foo` is the right way to do it.
# PEP 484 & PEP 585 Fun
if sys.version_info >= (3, 9):
    from builtins import dict as Dict
    from builtins import list as List
    from builtins import set as Set
    from builtins import tuple as Tuple
else:
    from typing import Dict, List, Set, Tuple

AdjacencyMatrix = Union[np.ndarray, sp.csr_matrix]

GraphRepresentation = Union[np.ndarray, sp.csr_matrix, nx.Graph]

RngType = Optional[Union[int, np.integer, np.random.Generator]]

__all__ = [
    "AdjacencyMatrix",
    "Dict",
    "List",
    "GraphRepresentation",
    "RngType",
    "Set",
    "Tuple",
]
