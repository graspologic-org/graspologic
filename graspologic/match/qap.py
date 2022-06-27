# adapted from scipy.optimze.quadratic_assignment()
# will live here temporalily until this function is officially released
# original code can be found here
# https://github.com/scipy/scipy/blob/master/scipy/optimize/_qap.py

import numbers
import operator
from typing import Any, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult, linear_sum_assignment
from typing_extensions import Literal

from graspologic.types import Dict, Tuple


def quadratic_assignment(
    A: np.ndarray,
    B: np.ndarray,
    method: Literal["faq"] = "faq",
    options: Optional[Dict[str, Any]] = None,
) -> OptimizeResult:
    r"""
    Approximates solution to the quadratic assignment problem and
    the graph matching problem.
    Quadratic assignment solves problems of the following form:
    .. math::
        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\
    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.
    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.
    Note that the quadratic assignment problem is NP-hard, is not
    known to be solvable in polynomial time, and is computationally
    intractable. Therefore, the results given are approximations,
    not guaranteed to be exact solutions.
    Parameters
    ----------
    A : 2d-array, square
        The square matrix :math:`A` in the objective function above.
    B : 2d-array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq'} (default: 'faq')
        The algorithm used to solve the problem.
        :ref:`'faq' <optimize.qap-faq>` (default) and
    options : dict, optional
        A dictionary of solver options. All solvers support the following:
        partial_match : 2d-array of integers, optional, (default = None)
            Allows the user to fix part of the matching between the two
            matrices. In the literature, a partial match is also
            known as a "seed" [2]_.
            Each row of `partial_match` specifies the indices of a pair of
            corresponding nodes, that is, node ``partial_match[i, 0]`` of `A`
            is matched to node ``partial_match[i, 1]`` of `B`. Accordingly,
            ``partial_match`` is an array of size ``(m , 2)``, where ``m`` is
            not greater than the number of nodes.
        maximize : bool (default = False)
            Setting `maximize` to ``True`` solves the Graph Matching Problem
            (GMP) rather than the Quadratic Assingnment Problem (QAP).
        rng : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
            This parameter defines the object to use for drawing random
            variates.
            If `rng` is ``None`` the `~np.random.RandomState` singleton is
            used.
            If `rng` is an int, a new ``RandomState`` instance is used,
            seeded with `rng`.
            If `rng` is already a ``RandomState`` or ``Generator``
            instance, then that object is used.
            Default is None.
        For method-specific options, see
        :func:`show_options('quadratic_assignment') <show_options>`.
    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` containing the following
        fields.
        col_ind : 1-D array
            An array of column indices corresponding with the best
            permutation of the nodes of `B` found.
        fun : float
            The corresponding value of the objective function.
        nit : int
            The number of iterations performed during optimization.
    Notes
    -----
    The default method :ref:`'faq' <optimize.qap-faq>` uses the Fast
    Approximate QAP algorithm [1]_; it is typically offers the best
    combination of speed and accuracy.
    Method :ref:`'2opt' <optimize.qap-2opt>` can be computationally expensive,
    but may be a useful alternative, or it can be used to refine the solution
    returned by another method.
    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`
    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`
    .. [3] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import quadratic_assignment
    >>> A = np.array([[0, 80, 150, 170], [80, 0, 130, 100],
    ...              [150, 130, 0, 120], [170, 100, 120, 0]])
    >>> B = np.array([[0, 5, 2, 7], [0, 0, 3, 8],
    ...              [0, 0, 0, 3], [0, 0, 0, 0]])
    >>> res = quadratic_assignment(A, B)
    >>> print(res)
     col_ind: array([0, 3, 2, 1])
         fun: 3260
         nit: 9
    The see the relationship between the returned ``col_ind`` and ``fun``,
    use ``col_ind`` to form the best permutation matrix found, then evaluate
    the objective function :math:`f(P) = trace(A^T P B P^T )`.
    >>> n = A.shape[0]
    >>> perm = res['col_ind']
    >>> P = np.eye(n, dtype=int)[perm]
    >>> fun = np.trace(A.T @ P @ B @ P.T)
    >>> print(fun)
    3260
    Alternatively, to avoid constructing the permutation matrix explicitly,
    directly permute the rows and columns of the distance matrix.
    >>> fun = np.trace(A.T @ B[perm][:, perm])
    >>> print(fun)
    3260
    Although not guaranteed in general, ``quadratic_assignment`` happens to
    have found the globally optimal solution.
    >>> from itertools import permutations
    >>> perm_opt, fun_opt = None, np.inf
    >>> for perm in permutations([0, 1, 2, 3]):
    ...     perm = np.array(perm)
    ...     fun = np.trace(A.T @ B[perm][:, perm])
    ...     if fun < fun_opt:
    ...         fun_opt, perm_opt = fun, perm
    >>> print(np.array_equal(perm_opt, res['col_ind']))
    True
    Here is an example for which the default method,
    :ref:`'faq' <optimize.qap-faq>`, does not find the global optimum.
    >>> A = np.array([[0, 5, 8, 6], [5, 0, 5, 1],
    ...              [8, 5, 0, 2], [6, 1, 2, 0]])
    >>> B = np.array([[0, 1, 8, 4], [1, 0, 5, 2],
    ...              [8, 5, 0, 5], [4, 2, 5, 0]])
    >>> res = quadratic_assignment(A, B)
    >>> print(res)
     col_ind: array([1, 0, 3, 2])
         fun: 178
         nit: 13
    If accuracy is important, consider using  :ref:`'2opt' <optimize.qap-2opt>`
    to refine the solution.
    >>> guess = np.array([np.arange(A.shape[0]), res.col_ind]).T
    >>> res = quadratic_assignment(A, B, method="2opt",
    ...                            options = {'partial_guess': guess})
    >>> print(res)
     col_ind: array([1, 2, 3, 0])
         fun: 176
         nit: 17
    """

    if options is None:
        options = {}

    method_key = method.lower()
    methods = {"faq": _quadratic_assignment_faq}
    if method_key not in methods:
        raise ValueError(f"method {method_key} must be in {list(methods.keys())}.")
    res = methods[method_key](A, B, **options)
    return res


def _calc_score(
    A: np.ndarray, B: np.ndarray, S: np.ndarray, perm: np.ndarray
) -> np.ndarray:
    # equivalent to objective function but avoids matmul
    return np.sum(A * B[perm][:, perm]) + np.sum(S[np.arange(len(S)), perm])


def _common_input_validation(
    A: np.ndarray, B: np.ndarray, partial_match: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    _partial_match = (
        partial_match if partial_match is not None else np.array([[], []]).T
    )

    _partial_match = np.atleast_2d(_partial_match).astype(int)

    msg = None
    if A.shape[0] != A.shape[1]:
        msg = "`A` must be square"
    elif B.shape[0] != B.shape[1]:
        msg = "`B` must be square"
    elif A.ndim != 2 or B.ndim != 2:
        msg = "`A` and `B` must have exactly two dimensions"
    elif A.shape != B.shape:
        msg = "`A` and `B` matrices must be of equal size"
    elif _partial_match.shape[0] > A.shape[0]:
        msg = "`partial_match` can have only as many seeds as there are nodes"
    elif _partial_match.shape[1] != 2:
        msg = "`partial_match` must have two columns"
    elif _partial_match.ndim != 2:
        msg = "`partial_match` must have exactly two dimensions"
    elif (_partial_match < 0).any():
        msg = "`partial_match` must contain only positive indices"
    elif (_partial_match >= len(A)).any():
        msg = "`partial_match` entries must be less than number of nodes"
    elif not len(set(_partial_match[:, 0])) == len(_partial_match[:, 0]) or not len(
        set(_partial_match[:, 1])
    ) == len(_partial_match[:, 1]):
        msg = "`partial_match` column entries must be unique"

    if msg is not None:
        raise ValueError(msg)

    return A, B, _partial_match


def _quadratic_assignment_faq(
    A: np.ndarray,
    B: np.ndarray,
    maximize: bool = False,
    partial_match: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    rng: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
    P0: Union[Literal["barycenter", "randomized"], np.ndarray] = "barycenter",
    shuffle_input: bool = False,
    maxiter: int = 30,
    tol: float = 0.03,
) -> OptimizeResult:
    r"""
    Solve the quadratic assignment problem (approximately).
    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the Fast Approximate QAP Algorithm
    (FAQ) [1]_.
    Quadratic assignment solves problems of the following form:
    .. math::
        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\
    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.
    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.
    Note that the quadratic assignment problem is NP-hard, is not
    known to be solvable in polynomial time, and is computationally
    intractable. Therefore, the results given are approximations,
    not guaranteed to be exact solutions.
    Parameters
    ----------
    A : 2d-array, square
        The square matrix :math:`A` in the objective function above.
    B : 2d-array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for 'faq'.
        :ref:`'2opt' <optimize.qap-2opt>` is also available.
    Options
    -------
    maximize : bool (default = False)
        Setting `maximize` to ``True`` solves the Graph Matching Problem (GMP)
        rather than the Quadratic Assingnment Problem (QAP). This is
        accomplished through trivial negation of the objective function.
    rng : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
        This parameter defines the object to use for drawing random
        variates.
        If `rng` is ``None`` the `~np.random.RandomState` singleton is
        used.
        If `rng` is an int, a new ``RandomState`` instance is used,
        seeded with `rng`.
        If `rng` is already a ``RandomState`` or ``Generator``
        instance, then that object is used.
        Default is None.
    partial_match : 2d-array of integers, optional, (default = None)
        Allows the user to fix part of the matching between the two
        matrices. In the literature, a partial match is also known as a
        "seed".
        Each row of `partial_match` specifies the indices of a pair of
        corresponding nodes, that is, node ``partial_match[i, 0]`` of `A` is
        matched to node ``partial_match[i, 1]`` of `B`. Accordingly,
        ``partial_match`` is an array of size ``(m , 2)``, where ``m`` is
        not greater than the number of nodes, :math:`n`.
    S : 2d-array, square
        A similarity matrix. Should be same shape as ``A`` and ``B``.   
        Note: the scale of `S` may effect the weight placed on the term 
        :math:`\\text{trace}(S^T P)` relative to :math:`\\text{trace}(A^T PBP^T)` 
        during the optimization process.
    P0 : 2d-array, "barycenter", or "randomized" (default = "barycenter")
        The initial (guess) permutation matrix or search "position"
        `P0`.
        `P0` need not be a proper permutation matrix;
        however, it must be :math:`m' x m'`, where :math:`m' = n - m`,
        and it must be doubly stochastic: each of its rows and columns must
        sum to 1.
        If unspecified or ``"barycenter"``, the non-informative "flat
        doubly stochastic matrix" :math:`J = 1*1^T/m'`, where :math:`1` is
        a :math:`m' \times 1` array of ones, is used. This is the "barycenter"
        of the search space of doubly-stochastic matrices.
        If ``"randomized"``, the algorithm will start from the
        randomized initial search position :math:`P_0 = (J + K)/2`,
        where :math:`J` is the "barycenter" and :math:`K` is a random
        doubly stochastic matrix.
    shuffle_input : bool (default = False)
        To avoid artificially high or low matching due to inherent
        sorting of input matrices, gives users the option
        to shuffle the nodes. Results are then unshuffled so that the
        returned results correspond with the node order of inputs.
        Shuffling may cause the algorithm to be non-deterministic,
        unless a random seed is set or an `rng` option is provided.
    maxiter : int, positive (default = 30)
        Integer specifying the max number of Franke-Wolfe iterations performed.
    tol : float (default = 0.03)
        A threshold for the stopping criterion. Franke-Wolfe
        iteration terminates when the change in search position between
        iterations is sufficiently small, that is, when the relative Frobenius
        norm, :math:`\frac{||P_{i}-P_{i+1}||_F}{\sqrt{len(P_{i})}} \leq tol`,
        where :math:`i` is the iteration number.
    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` containing the following
        fields.
        col_ind : 1-D array
            An array of column indices corresponding with the best
            permutation of the nodes of `B` found.
        fun : float
            The corresponding value of the objective function.
        nit : int
            The number of Franke-Wolfe iterations performed.
    Notes
    -----
    The algorithm may be sensitive to the initial permutation matrix (or
    search "position") due to the possibility of several local minima
    within the feasible region. A barycenter initialization is more likely to
    result in a better solution than a single random initialization. However,
    ``quadratic_assignment`` calling several times with different random
    initializations may result in a better optimum at the cost of longer
    total execution time.
    Examples
    --------
    As mentioned above, a barycenter initialization often results in a better
    solution than a single random initialization.
    >>> np.random.seed(0)
    >>> n = 15
    >>> A = np.random.rand(n, n)
    >>> B = np.random.rand(n, n)
    >>> res = quadratic_assignment(A, B)  # FAQ is default method
    >>> print(res.fun)
    46.871483385480545 # may vary
    >>> options = {"P0": "randomized"}  # use randomized initialization
    >>> res = quadratic_assignment(A, B, options=options)
    >>> print(res.fun)
    47.224831071310625 # may vary
    However, consider running from several randomized initializations and
    keeping the best result.
    >>> res = min([quadratic_assignment(A, B, options=options)
    ...            for i in range(30)], key=lambda x: x.fun)
    >>> print(res.fun)
    46.671852533681516 # may vary
    The '2-opt' method can be used to further refine the results.
    >>> options = {"partial_guess": np.array([np.arange(n), res.col_ind]).T}
    >>> res = quadratic_assignment(A, B, method="2opt", options=options)
    >>> print(res.fun)
    46.47160735721583 # may vary
    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`
    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`
    """

    maxiter = operator.index(maxiter)

    # ValueError check
    A, B, partial_match_value = _common_input_validation(A, B, partial_match)

    msg = None
    if isinstance(P0, str) and P0 not in {"barycenter", "randomized"}:
        msg = "Invalid 'P0' parameter string"
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    elif tol <= 0:
        msg = "'tol' must be a positive float"
    if msg is not None:
        raise ValueError(msg)

    if not isinstance(S, np.ndarray):
        raise ValueError("`S` must be an ndarray")
    elif S.shape != (S.shape[0], S.shape[0]):
        raise ValueError("`S` must be square")
    elif S.shape != A.shape:
        raise ValueError("`S`, `A`, and `B` matrices must be of equal size")
    else:
        s_value = S

    rng = check_random_state(rng)
    n = A.shape[0]  # number of vertices in graphs
    n_seeds = partial_match_value.shape[0]  # number of seeds
    n_unseed = n - n_seeds

    # check outlier cases
    if n == 0 or partial_match_value.shape[0] == n:
        # Cannot assume partial_match is sorted.
        sort_inds = np.argsort(partial_match_value[:, 0])
        partial_match_value = partial_match_value[sort_inds]
        score = _calc_score(A, B, s_value, partial_match_value[:, 1])
        res = {"col_ind": partial_match_value[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1

    nonseed_B = np.setdiff1d(range(n), partial_match_value[:, 1])
    perm_S = np.copy(nonseed_B)
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)
        # shuffle_input to avoid results from inputs that were already matched

    nonseed_A = np.setdiff1d(range(n), partial_match_value[:, 0])
    perm_A = np.concatenate([partial_match_value[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match_value[:, 1], nonseed_B])

    s_value = s_value[:, perm_B]

    # definitions according to Seeded Graph Matching [2].
    A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    S22 = s_value[perm_S, n_seeds:]

    # [1] Algorithm 1 Line 1 - choose initialization
    if isinstance(P0, str):
        # initialize J, a doubly stochastic barycenter
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        if P0 == "barycenter":
            P = J
        elif P0 == "randomized":
            # generate a nxn matrix where each entry is a random number [0, 1]
            # would use rand, but Generators don't have it
            # would use random, but old mtrand.RandomStates don't have it
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * 0.5 + K * 0.5
    elif isinstance(P0, np.ndarray):
        _P0 = np.atleast_2d(P0)
        _check_init_input(_P0, n_unseed)
        invert_inds = np.argsort(nonseed_B)
        perm_nonseed_B = np.argsort(invert_inds)
        P = _P0[:, perm_nonseed_B]
    else:
        msg = "`init` must either be of type str or np.ndarray."
        raise TypeError(msg)

    const_sum = A21 @ B21.T + A12.T @ B12 + S22

    # [1] Algorithm 1 Line 2 - loop while stopping criteria not met
    for n_iter in range(1, maxiter + 1):
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        grad_fp = const_sum + A22 @ P @ B22.T + A22.T @ P @ B22
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
        Q = np.eye(n_unseed)[cols]

        # [1] Algorithm 1 Line 5 - compute the step size
        # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
        # terms as ax**2 + bx + c. c does not affect location of minimum
        # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
        # apply where possible for efficiency.
        R = P - Q
        b21 = ((R.T @ A21) * B21).sum()
        b12 = ((R.T @ A12.T) * B12.T).sum()
        AR22 = A22.T @ R
        BR22 = B22 @ R.T
        b22a = (AR22 * B22.T[cols]).sum()
        b22b = (A22 * BR22[cols]).sum()
        s = (S22 * R).sum()
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b + s
        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])

        # [1] Algorithm 1 Line 6 - Update P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    # [1] Algorithm 1 Line 7 - end main loop

    # [1] Algorithm 1 Line 8 - project onto the set of permutation matrices
    _, col = linear_sum_assignment(-P)
    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]

    score = _calc_score(A, B, s_value, unshuffled_perm)

    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter}

    return OptimizeResult(res)


def _check_init_input(P0: np.ndarray, n: int) -> None:
    row_sum = np.sum(P0, axis=0)
    col_sum = np.sum(P0, axis=1)
    tol = 1e-3
    msg = None
    if P0.shape != (n, n):
        msg = "`P0` matrix must have shape m' x m', where m'=n-m"
    elif (
        (~np.isclose(row_sum, 1, atol=tol)).any()
        or (~np.isclose(col_sum, 1, atol=tol)).any()
        or (P0 < 0).any()
    ):
        msg = "`P0` matrix must be doubly stochastic"
    if msg is not None:
        raise ValueError(msg)


def _split_matrix(
    X: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # definitions according to Seeded Graph Matching [2].
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _doubly_stochastic(P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


# copy-pasted from scipy scipy._lib._util
# which was copy-pasted from scikit-learn utils/validation.py
# this was just modified to add proper typing for returns
# also, shouldn't have been importing private function from scipy anyway
def check_random_state(
    seed: Union[None, int, np.random.RandomState, np.random.Generator]
) -> Union[np.random.RandomState, np.random.Generator]:
    """Turn seed into a np.random.RandomState instance

    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    If seed is a new-style np.random.Generator, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        int_seed: int = int(seed)  # necessary for typing/mypy
        return np.random.RandomState(int_seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )
