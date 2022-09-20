# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, NamedTuple, Optional

import numpy as np
from beartype import beartype
from joblib import Parallel, delayed
from sklearn.utils import check_scalar

from graspologic.match.solver import _GraphMatchSolver
from graspologic.types import Dict, List, RngType

from .types import (
    AdjacencyMatrix,
    Int,
    MultilayerAdjacency,
    PaddingType,
    PartialMatchType,
    Scalar,
)


class MatchResult(NamedTuple):
    indices_A: np.ndarray
    """
    Sorted indices in ``A`` which were matched.
    """

    indices_B: np.ndarray
    """
    Indices in ``B`` which were matched. Element ``indices_B[i]`` was matched
    to element ``indices_A[i]``. ``indices_B`` can also be thought of as a 
    permutation of the nodes of ``B`` with respect to ``A``.
    """

    score: float
    """
    Objective function value at the end of optimization.
    """

    misc: List[Dict[str, Any]]
    """
    List of length ``n_init`` containing information about each run. Fields for
    each run are ``score``, ``n_iter``, ``convex_solution``, and ``converged``.
    """


@beartype
def graph_match(
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    AB: Optional[MultilayerAdjacency] = None,
    BA: Optional[MultilayerAdjacency] = None,
    S: Optional[AdjacencyMatrix] = None,
    partial_match: Optional[PartialMatchType] = None,
    init: Optional[np.ndarray] = None,
    init_perturbation: Scalar = 0.0,
    n_init: Int = 1,
    shuffle_input: bool = True,
    maximize: bool = True,
    padding: PaddingType = "naive",
    n_jobs: Optional[Int] = None,
    max_iter: Int = 30,
    tol: Scalar = 0.01,
    verbose: Int = 0,
    rng: Optional[RngType] = None,
    transport: bool = False,
    transport_regularizer: Scalar = 100,
    transport_tol: Scalar = 5e-2,
    transport_max_iter: Int = 1000,
) -> MatchResult:
    """
    Attempts to solve the Graph Matching Problem or the Quadratic Assignment Problem
    (QAP) through an implementation of the Fast Approximate QAP (FAQ) Algorithm [1].

    This algorithm can be thought of as finding an alignment of the vertices of two
    graphs which minimizes the number of induced edge disagreements, or, in the case
    of weighted graphs, the sum of squared differences of edge weight disagreements.
    Various extensions to the original FAQ algorithm are also included in this function
    ([2-5]).


    Parameters
    ----------
    A : {ndarray, csr_matrix, csr_array} of shape (n, n), or a list thereof
        The first (potentially multilayer) adjacency matrix to be matched. Multiplex
        networks (e.g. a network with multiple edge types) can be used by inputting a
        list of the adjacency matrices for each edge type.

    B : {ndarray, csr_matrix, csr_array} of shape (m, m), or a list thereof
        The second (potentially multilayer) adjacency matrix to be matched. Must have
        the same number of layers as ``A``, but need not have the same size
        (see ``padding``).

    AB : {ndarray, csr_matrix, csr_array} of shape (n, m), or a list thereof, default=None
        A (potentially multilayer) matrix representing connections from the objects
        indexed in ``A`` to those in ``B``, used for bisected graph matching (see [2]).

    BA : {ndarray, csr_matrix, csr_array} of shape (m, n), or a list thereof, default=None
        A (potentially multilayer) matrix representing connections from the objects
        indexed in ``B`` to those in ``A``, used for bisected graph matching (see [2]).

    S : {ndarray, csr_matrix, csr_array} of shape (n, m), default=None
        A matrix representing the similarity of objects indexed in ``A`` to each object
        indexed in ``B``. Note that the scale (i.e. the norm) of this matrix will affect
        how strongly the similarity (linear) term is weighted relative to the adjacency
        (quadratic) terms.

    partial_match : ndarray of shape (n_matches, 2), dtype=int, or tuple of two array-likes of shape (n_matches,), default=None
        Indices specifying known matches to include in the optimization. The
        first column represents indices of the objects in ``A``, and the second column
        represents their corresponding matches in ``B``.

    init : ndarray of shape (n_unseed, n_unseed), default=None
        Initialization for the algorithm. Setting to None specifies the "barycenter",
        which is the most commonly used initialization and
        represents an uninformative (flat) initialization. If a ndarray, then this
        matrix must be square and have size equal to the number of unseeded (not
        already matched in ``partial_match``) nodes.

    init_perturbation : float, default=0.0
        Weight of the random perturbation from ``init`` that the initialization will
        undergo. Must be between 0 and 1.

    n_init : int, default=1
        Number of initializations/runs of the algorithm to repeat. The solution with
        the best objective function value over all initializations is kept. Increasing
        ``n_init`` can improve performance but will take longer.

    shuffle_input : bool, default=True
        Whether to shuffle the order of the inputs internally during optimization. This
        option is recommended to be kept to True besides for testing purposes; it
        alleviates a dependence of the solution on the (arbitrary) ordering of the
        input rows/columns.

    maximize : bool, default=True
        Whether to maximize the objective function (graph matching problem) or minimize
        it (quadratic assignment problem). ``maximize=True`` corresponds to trying to
        find a permutation wherein the input matrices are as similar as possible - for
        adjacency matrices, this corresponds to maximizing the overlap of the edges of
        the two networks. Conversely, ``maximize=False`` would attempt to make this
        overlap as small as possible.

    padding : {"naive", "adopted"}, default="naive"
        Specification of a padding scheme if ``A`` and ``B`` are not of equal size. See
        the `padded graph matching tutorial <https://microsoft.github.io/graspologic/tutorials/matching/padded_gm.html>`_
        or [3] for more explanation. Adopted padding has not been tested for weighted
        networks; use with caution.

    n_jobs : int, default=None
        The number of jobs to run in parallel. Parallelization is over the
        initializations, so only relevant when ``n_init > 1``. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors. See
        :class:`joblib.Parallel` for more details.

    max_iter : int, default=30
        Must be 1 or greater, specifying the max number of iterations for the algorithm.
        Setting this value higher may provide more precise solutions at the cost of
        longer computation time.

    tol : float, default=0.01
        Stopping tolerance for the FAQ algorithm. Setting this value smaller may provide
        more precise solutions at the cost of longer computation time.

    verbose : int, default=0
        A positive number specifying the level of verbosity for status updates in the
        algorithm's progress. If ``n_jobs`` > 1, then this parameter behaves as the
        ``verbose`` parameter for :class:`joblib.Parallel`. Otherwise, will print
        increasing levels of information about the algorithm's progress for each
        initialization.

    rng : int or np.random.Generator, default=None
        Allows the specification of a random seed (positive integer) or a
        :class:`np.random.Generator` object to ensure reproducibility.

    transport : bool, default=False
        Whether to enable use of regularized optimal transport for determining the step
        direction as described in [4]. May improve accuracy/speed, especially for large
        inputs and data where the correlation between edges is not close to 1.

    transport_regularizer : int or float, default=100
        Strength of the entropic regularization in the optimal transport solver.

    transport_tol : int or float, default=0.05,
        Must be positive. Stopping tolerance for the optimal transport solver. Setting
        this value smaller may provide more precise solutions at the cost of longer
        computation time.

    transport_max_iter : int, default=1000
        Must be positive. Maximum number of iterations for the optimal transport solver.
        Setting this value higher may provide more precise solutions at the cost of
        longer computation time.

    Returns
    -------
    res: MatchResult
        ``MatchResult`` containing the following fields.

        indices_A : ndarray
            Sorted indices in ``A`` which were matched.

        indices_B : ndarray
            Indices in ``B`` which were matched. Element ``indices_B[i]`` was matched
            to element ``indices_A[i]``. ``indices_B`` can also be thought of as a
            permutation of the nodes of ``B`` with respect to ``A``.

        score : float
            Objective function value at the end of optimization.

        misc : list of dict
            List of length ``n_init`` containing information about each run. Fields for
            each run are ``score``, ``n_iter``, ``convex_solution``, and ``converged``.

    Notes
    -----
    Many extensions [2-5] to the original FAQ algorithm are included in this function.
    The full objective function which this function aims to solve can be written as

    .. math:: f(P) = - \sum_{k=1}^K \|A^{(k)} - PB^{(k)}P^T\|_F^2 - \sum_{k=1}^K \|(AB)^{(k)}P^T - P(BA)^{(k)}\|_F^2 + trace(SP^T)

    where :math:`P` is a permutation matrix we are trying to learn, :math:`A^{(k)}` is the adjacency
    matrix in network :math:`A` for the :math:`k`-th edge type (and likewise for B), :math:`(AB)^{(k)}`
    (with a slight abuse of notation, but for consistency with the code) is an adjacency
    matrix representing a subgraph of any connections which go from objects in :math:`A` to
    those in :math:`B` (and defined likewise for :math:`(BA)`), and :math:`S` is a
    similarity matrix indexing the similarity of each object in :math:`A` to each object
    in :math:`B`.

    If ``partial_match`` is used, then the above will be maximized/minimized over the
    set of permutations which respect this partial matching of the two networks.

    If ``maximize``, this function will attempt to maximize :math:`f(P)` (solve the graph
    matching problem); otherwise, it will be minimized.

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik, S.G. Kratzer,
        E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and C.E. Priebe, “Fast
        approximate quadratic programming for graph matching,” PLOS one, vol. 10,
        no. 4, p. e0121002, 2015.

    .. [2] B.D. Pedigo, M. Winding, C.E. Priebe, J.T. Vogelstein, "Bisected graph
        matching improves automated pairing of bilaterally homologous neurons from
        connectomes," bioRxiv 2022.05.19.492713 (2022)

    .. [3] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski, C. Priebe,
        "Seeded graph matching," Pattern Recognit. 87 (2019) 203–215

    .. [4] A. Saad-Eldin, B.D. Pedigo, C.E. Priebe, J.T. Vogelstein "Graph Matching via
       Optimal Transport," arXiv 2111.05366 (2021)

    .. [5] K. Pantazis, D.L. Sussman, Y. Park, Z. Li, C.E. Priebe, V. Lyzinski,
       "Multiplex graph matching matched filters," Applied Network Science (2022)
    """

    max_seed = np.iinfo(np.uint32).max

    if (rng is not None) and (not isinstance(rng, np.random.Generator)):
        check_scalar(rng, "rng", (int, np.integer), min_val=0, max_val=max_seed)
    # otherwise the input is None or a random Generator - these can be passed in to
    # default_rng safely

    rng = np.random.default_rng(rng)

    seeds = rng.integers(max_seed, size=n_init)

    if n_init > 1:
        parallel_verbose = verbose
        solver_verbose: Int = 0
    else:
        parallel_verbose = 0
        solver_verbose = verbose

    solver = _GraphMatchSolver(
        A=A,
        B=B,
        AB=AB,
        BA=BA,
        S=S,
        partial_match=partial_match,
        init=init,
        init_perturbation=init_perturbation,
        verbose=solver_verbose,
        shuffle_input=shuffle_input,
        padding=padding,
        maximize=maximize,
        max_iter=max_iter,
        tol=tol,
        transport=transport,
        transport_regularizer=transport_regularizer,
        transport_tol=transport_tol,
        transport_max_iter=transport_max_iter,
    )

    def run_single_graph_matching(seed: RngType) -> MatchResult:
        solver.solve(seed)
        matching = solver.matching_
        indices_A = matching[:, 0]
        indices_B = matching[:, 1]
        score = solver.score_
        misc: Dict[str, Any] = {}
        misc["score"] = score
        misc["n_iter"] = solver.n_iter_
        misc["convex_solution"] = solver.convex_solution_
        misc["converged"] = solver.converged_
        return MatchResult(indices_A, indices_B, score, [misc])

    seeds = rng.integers(max_seed, size=n_init)
    parallel = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)
    results = parallel(delayed(run_single_graph_matching)(seed) for seed in seeds)

    # get the indices for the best run
    best_func = max if maximize else min
    best_result = best_func(results, key=lambda x: x.score)

    # also collate various extra info about all of the runs
    miscs = [x.misc[0] for x in results]

    return MatchResult(
        best_result.indices_A, best_result.indices_B, best_result.score, miscs
    )
