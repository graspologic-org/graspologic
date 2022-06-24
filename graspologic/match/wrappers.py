from typing import Literal, Optional, Union

import numpy as np

from graspologic.types import AdjacencyMatrix, Dict, List, Tuple

# Type aliases
PaddingType = Literal["adopted", "naive"]
InitMethodType = Literal["barycenter", "rand", "randomized"]
RandomStateType = Optional[Union[int, np.random.RandomState, np.random.Generator]]
ArrayLikeOfIndexes = Union[List[int], np.ndarray]
MultilayerAdjacency = Union[List[AdjacencyMatrix], AdjacencyMatrix, np.ndarray]
Scalar = Union[int, float, np.integer]
Int = Union[int, np.integer]


def graph_match(
    A,
    B,
    partial_match=None,
    similarity=None,
    AtoB=None,
    BtoA=None,
    init="barycenter",
    n_init=1,
    shuffle_input=True,
    maximize=True,
    padding="adopted",
    n_jobs=None,
    max_iter=30,
    tol=0.01,
    verbose=0,
    random_state=None,
    use_numba=True,
    transport=False,
    transport_regularizer: Scalar = 100,
    transport_tolerance: Scalar = 5e-2,
    transport_max_iter: Int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    pass


def soft_graph_match(A, B, ...)