from collections import namedtuple
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_scalar
from typing_extensions import Literal

from graspologic.match.solver import GraphMatchSolver
from graspologic.types import AdjacencyMatrix, Dict, List, RngType, Tuple

# Type aliases
PaddingType = Literal["adopted", "naive"]
# InitMethodType = Union[float, Literal["barycenter", "random"]]
ArrayLikeOfIndexes = Union[List[int], np.ndarray]
MultilayerAdjacency = Union[List[AdjacencyMatrix], AdjacencyMatrix, np.ndarray]
Scalar = Union[int, float, np.integer]
Int = Union[int, np.integer]
InitType = Union[Literal["barycenter"], np.ndarray]

MatchResult = namedtuple("MatchResult", ["indices_A", "indices_B", "score", "misc"])


def graph_match(
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    partial_match: Optional[np.ndarray] = None,
    similarity: Optional[AdjacencyMatrix] = None,
    AtoB: Optional[MultilayerAdjacency] = None,
    BtoA: Optional[MultilayerAdjacency] = None,
    init: InitType = "barycenter",
    init_perturbation: Scalar = 0.0,
    n_init: Int = 1,
    shuffle_input: bool = True,
    maximize: bool = True,
    padding: PaddingType = "adopted",
    n_jobs: Optional[Int] = None,
    maxiter: Int = 30,
    tol: Scalar = 0.01,
    verbose: Int = 0,
    rng: Optional[RngType] = None,
    use_numba: bool = False,
    transport: bool = False,
    transport_regularizer: Scalar = 100,
    transport_tolerance: Scalar = 5e-2,
    transport_maxiter: Int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:

    if use_numba:
        raise NotImplementedError("Still working on numba implementation")

    max_seed = np.iinfo(np.uint32).max

    if (rng is not None) and (not isinstance(rng, np.random.Generator)):
        rng = check_scalar(rng, "rng", (int, np.integer), min_val=0, max_val=max_seed)
    # otherwise the input is None or a random Generator - these can be passed in to
    # default_rng safely

    rng = np.random.default_rng(rng)

    seeds = rng.integers(max_seed, size=n_init)

    if n_init > 1:
        parallel_verbose = verbose
        solver_verbose = 0
    else:
        parallel_verbose = 0
        solver_verbose = verbose

    solver = GraphMatchSolver(
        A=A,
        B=B,
        AB=AtoB,
        BA=BtoA,
        similarity=similarity,
        partial_match=partial_match,
        init=init,
        verbose=solver_verbose,
        shuffle_input=shuffle_input,
        padding=padding,
        maximize=maximize,
        maxiter=maxiter,
        tol=tol,
        use_numba=use_numba,
        transport=transport,
        transport_regularizer=transport_regularizer,
        transport_tolerance=transport_tolerance,
        transport_maxiter=transport_maxiter,
    )

    def run_single_graph_matching(seed):
        solver.solve(seed)
        matching = solver.matching_
        indices_A = matching[:, 0]
        indices_B = matching[:, 1]
        score = solver.score_
        misc = {}
        misc["score"] = score
        misc["n_iter"] = solver.n_iter_
        misc["convex_solution"] = solver.convex_solution_
        misc["converged"] = solver.converged_
        return MatchResult(indices_A, indices_B, score, misc)

    seeds = rng.integers(max_seed, size=n_init)
    parallel = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)
    results = parallel(delayed(run_single_graph_matching)(seed) for seed in seeds)
    # results = [run_single_graph_matching(seeds)]

    # get the indices for the best run
    best_func = max if maximize else min
    best_result = best_func(results, key=lambda x: x.score)

    # also collate various extra info about all of the runs
    miscs = [x.misc for x in results]
    misc_df = pd.DataFrame(miscs)

    return MatchResult(
        best_result.indices_A, best_result.indices_B, best_result.score, misc_df
    )
