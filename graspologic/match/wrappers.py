from typing import Optional, Union

import numpy as np
from typing_extensions import Literal

from graspologic.types import AdjacencyMatrix, Dict, List, Tuple

# Type aliases
PaddingType = Literal["adopted", "naive"]
InitMethodType = Union[float, Literal["barycenter", "random"]]
RandomStateType = Optional[
    Union[int, np.integer, np.random.RandomState, np.random.Generator]
]
ArrayLikeOfIndexes = Union[List[int], np.ndarray]
MultilayerAdjacency = Union[List[AdjacencyMatrix], AdjacencyMatrix, np.ndarray]
Scalar = Union[int, float, np.integer]
Int = Union[int, np.integer]


def graph_match(
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    partial_match: Optional[np.ndarray] = None,
    similarity: Optional[AdjacencyMatrix] = None,
    AtoB: Optional[MultilayerAdjacency] = None,
    BtoA: Optional[MultilayerAdjacency] = None,
    init: InitMethodType = "barycenter",
    n_init: Int = 1,
    shuffle_input: bool = True,
    maximize: bool = True,
    padding: PaddingType = "adopted",
    n_jobs: Optional[Int] = None,
    maxiter: Int = 30,
    tol: Scalar = 0.01,
    verbose: Int = 0,
    random_state: Optional[RandomStateType] = None,
    numba: bool = True,
    transport: bool = False,
    transport_regularizer: Scalar = 100,
    transport_tolerance: Scalar = 5e-2,
    transport_max_iter: Int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    pass
