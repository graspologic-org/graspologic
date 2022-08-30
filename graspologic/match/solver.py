# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import time
import warnings
from functools import wraps
from typing import Callable, Optional, Union

import numpy as np
from beartype import beartype
from ot import sinkhorn
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.utils import check_scalar

from graspologic.types import List, RngType, Tuple

from .types import (
    AdjacencyMatrix,
    Int,
    MultilayerAdjacency,
    PaddingType,
    PartialMatchType,
    Scalar,
    csr_array,
)


def parameterized(dec: Callable) -> Callable:
    def layer(*args, **kwargs) -> Callable:  # type: ignore
        def repl(f: Callable) -> Callable:
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parameterized
def write_status(f: Callable, msg: str, level: int) -> Callable:
    @wraps(f)
    def wrap(*args, **kw):  # type: ignore
        obj = args[0]
        verbose = obj.verbose
        if level <= verbose:
            total_msg = (level - 1) * "   "
            total_msg += obj.status() + " " + msg
            print(total_msg)
        if (verbose >= 4) and (level <= verbose):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            sec = te - ts
            output = total_msg + f" took {sec:.3f} seconds."
            print(output)
        else:
            result = f(*args, **kw)
        return result

    return wrap


class _GraphMatchSolver:
    @beartype
    def __init__(
        self,
        A: MultilayerAdjacency,
        B: MultilayerAdjacency,
        AB: Optional[MultilayerAdjacency] = None,
        BA: Optional[MultilayerAdjacency] = None,
        S: Optional[AdjacencyMatrix] = None,
        partial_match: Optional[PartialMatchType] = None,
        init: Optional[np.ndarray] = None,
        init_perturbation: Scalar = 0.0,
        verbose: Int = False,
        shuffle_input: bool = True,
        padding: PaddingType = "naive",
        maximize: bool = True,
        max_iter: Int = 30,
        tol: Scalar = 0.03,
        transport: bool = False,
        transport_regularizer: Scalar = 100,
        transport_tol: Scalar = 5e-2,
        transport_max_iter: Int = 1000,
    ):
        # TODO check if init is doubly stochastic
        self.init = init
        check_scalar(
            init_perturbation,
            name="init_perturbation",
            target_type=(float, int),
            min_val=0,
            max_val=1,
        )
        self.init_perturbation = init_perturbation
        self.verbose = verbose
        self.shuffle_input = shuffle_input
        self.maximize = maximize
        check_scalar(max_iter, name="max_iter", target_type=int, min_val=1)
        self.max_iter = max_iter
        check_scalar(tol, name="tol", target_type=(int, float), min_val=0)
        self.tol = tol
        self.padding = padding

        self.transport = transport
        self.transport_regularizer = transport_regularizer
        check_scalar(
            transport_tol, name="transport_tol", target_type=(int, float), min_val=0
        )
        self.transport_tol = transport_tol
        check_scalar(
            transport_max_iter, name="transport_max_iter", target_type=int, min_val=1
        )
        self.transport_max_iter = transport_max_iter

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

        # convert everything to make sure they are 3D arrays (first dim is layer)
        A = _check_input_matrix(A, n_layers=None)
        self.n_layers = len(A)
        B = _check_input_matrix(B, n_layers=self.n_layers)
        if len(A) != len(B):
            raise ValueError("`A` and `B` must have same number of layers.")

        # get some useful sizes
        self.n_A = A[0].shape[0]
        self.n_B = B[0].shape[0]

        if partial_match is None:
            self._seeded = False
        else:
            self._seeded = True

        seeds = _check_partial_match(partial_match, self.n_A, self.n_B)

        n_seeds = len(seeds)
        self.n_seeds = n_seeds

        if (self.n_A != self.n_B) and padding == "adopted":
            contra_sparse = False
        else:
            contra_sparse = True
        # check for between-graph terms
        # default to sparse if possible since all 0s
        if AB is None:
            if contra_sparse:
                AB = self.n_layers * [csr_array((self.n_A, self.n_B))]
            else:
                AB = self.n_layers * [np.zeros((self.n_A, self.n_B))]
        else:
            AB = _check_input_matrix(AB, n_layers=self.n_layers)
        if BA is None:
            if contra_sparse:
                BA = self.n_layers * [csr_array((self.n_B, self.n_A))]
            else:
                BA = self.n_layers * [np.zeros((self.n_B, self.n_A))]
        else:
            BA = _check_input_matrix(BA, n_layers=self.n_layers)

        # check all input dims
        # can safely assume that the first matrix in each is representative of dimension
        # because of check done in _check_input_matrix
        _compare_dimensions(A, A, "row", "column", "A", "A")
        _compare_dimensions(B, B, "row", "column", "B", "B")
        _compare_dimensions(A, AB, "row", "row", "A", "AB")
        _compare_dimensions(B, AB, "row", "column", "B", "AB")
        _compare_dimensions(A, BA, "row", "column", "A", "BA")
        _compare_dimensions(B, BA, "row", "row", "B", "BA")

        # padding for unequally sized inputs
        if self.n_A != self.n_B:
            self.n = np.max((self.n_A, self.n_B))
            A = _multilayer_adj_pad(A, n_padded=self.n, method=self.padding)
            B = _multilayer_adj_pad(B, n_padded=self.n, method=self.padding)
            AB = _multilayer_adj_pad(AB, n_padded=self.n, method=self.padding)
            BA = _multilayer_adj_pad(BA, n_padded=self.n, method=self.padding)
            self.padded = True
            if self.n_A > self.n_B:
                self._padded_B = True
            else:
                self._padded_B = False
        else:
            self.padded = False
            self.n = self.n_A

        # check for similarity term
        if S is None:
            S = csr_array((self.n, self.n))

        _compare_dimensions(A, [S], "row", "row", "A", "S")
        _compare_dimensions(B, [S], "row", "column", "B", "S")

        self.A = A
        self.B = B
        self.AB = AB
        self.BA = BA
        self.S = S

        # set up so that seeds are first and we can grab subgraphs easily
        # TODO could also do this slightly more efficiently just w/ smart indexing?
        sort_inds = np.argsort(seeds[:, 0])
        seeds = seeds[sort_inds]
        self.seeds = seeds
        nonseed_A = np.setdiff1d(np.arange(self.n), seeds[:, 0])
        nonseed_B = np.setdiff1d(np.arange(self.n), seeds[:, 1])
        perm_A = np.concatenate([seeds[:, 0], nonseed_A])
        perm_B = np.concatenate([seeds[:, 1], nonseed_B])
        self.perm_A = perm_A
        self.perm_B = perm_B
        self._undo_perm_A = np.argsort(perm_A)
        self._undo_perm_B = np.argsort(perm_B)

        # permute each (sub)graph appropriately
        A = _permute_multilayer(A, perm_A, rows=True, columns=True)
        B = _permute_multilayer(B, perm_B, rows=True, columns=True)
        AB = _permute_multilayer(AB, perm_A, rows=True, columns=False)
        AB = _permute_multilayer(AB, perm_B, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_A, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_B, rows=True, columns=False)
        S = S[perm_A][:, perm_B]

        # split into subgraphs of seed-to-seed (ss), seed-to-nonseed (sn), etc.
        # main thing being permuted has no subscript
        self.A_ss, self.A_sn, self.A_ns, self.A_nn = _split_multilayer_matrix(
            A, n_seeds
        )
        self.B_ss, self.B_sn, self.B_ns, self.B_nn = _split_multilayer_matrix(
            B, n_seeds
        )
        self.AB_ss, self.AB_sn, self.AB_ns, self.AB_nn = _split_multilayer_matrix(
            AB, n_seeds
        )
        self.BA_ss, self.BA_sn, self.BA_ns, self.BA_nn = _split_multilayer_matrix(
            BA, n_seeds
        )

        self.n_unseed = self.B_nn[0].shape[0]

        self.S_ss, self.S_sn, self.S_ns, self.S_nn = _split_matrix(S, n_seeds)

    def solve(self, rng: RngType = None) -> None:
        rng = np.random.default_rng(rng)

        self.n_iter_ = 0
        if self.n_seeds == self.n:  # all seeded, break
            P = np.empty((0, 0))
            self.converged_ = True
        else:
            P = self.initialize(rng)
            self.compute_constant_terms()
            for n_iter in range(self.max_iter):
                self.n_iter_ = n_iter + 1

                gradient = self.compute_gradient(P)
                Q = self.compute_step_direction(gradient, rng)
                alpha = self.compute_step_size(P, Q)

                # take a step in this direction
                P_new = alpha * P + (1 - alpha) * Q

                if self.check_converged(P, P_new):
                    self.converged_ = True
                    P = P_new
                    break
                P = P_new

        self.finalize(P, rng)

    @write_status("Initializing", 1)
    def initialize(self, rng: np.random.Generator) -> np.ndarray:
        # user custom initialization
        if isinstance(self.init, np.ndarray):
            _check_init_input(self.init, self.n_unseed)
            J = self.init
        # else, just a flat, uninformative initializaiton, also called the barycenter
        # (of the set of doubly stochastic matrices)
        else:
            n_unseed = self.n_unseed
            J = np.full((n_unseed, n_unseed), 1 / n_unseed)

        if self.init_perturbation > 0:
            # create a random doubly stochastic matrix
            # TODO unsure if this is actually uniform over the Birkoff polytope
            # I suspect it is not. Not even sure if humankind knows how to generate such
            # a matrix efficiently...

            # start with random (uniform 0-1 values) matrix
            K = rng.uniform(size=(n_unseed, n_unseed))
            # use Sinkhorn algo. to project to closest doubly stochastic
            K = _doubly_stochastic(K)

            # to a convex combination with either barycenter or input initialization
            P = J * (1 - self.init_perturbation) + K * (self.init_perturbation)
        else:
            P = J

        self.converged_ = False
        return P

    @write_status("Computing constant terms", 2)
    def compute_constant_terms(self) -> None:
        self.constant_sum = np.zeros((self.n_unseed, self.n_unseed))
        if self._seeded:
            n_layers = len(self.A_nn)
            for i in range(n_layers):
                self.constant_sum += (
                    self.A_ns[i] @ self.B_ns[i].T  # ipsi
                    + self.A_sn[i].T @ self.B_sn[i]  # ipsi
                    + self.AB_ns[i] @ self.BA_ns[i].T  # contra
                    + self.BA_sn[i].T @ self.AB_sn[i]  # contra
                )
        self.constant_sum += self.S_nn

    @write_status("Computing gradient", 2)
    def compute_gradient(self, P: np.ndarray) -> np.ndarray:
        gradient = _compute_gradient(
            P, self.A_nn, self.B_nn, self.AB_nn, self.BA_nn, self.constant_sum
        )
        return gradient

    @write_status("Solving assignment problem", 2)
    def compute_step_direction(
        self, gradient: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        if self.transport:
            Q = self.linear_sum_transport(gradient)
        else:
            permutation = self.linear_sum_assignment(gradient, rng)
            Q = np.eye(self.n_unseed)[permutation]
        return Q

    def linear_sum_assignment(
        self, P: np.ndarray, rng: np.random.Generator, maximize: Optional[bool] = None
    ) -> np.ndarray:
        """This is a modified version of LAP which (in expectation) does not care
        about the order of the inputs. This matters because scipy LAP settles ties
        (which do come up) based on the ordering of the inputs. This can lead to
        artificially high matching accuracy when the user inputs data which is in the
        correct permutation, for example."""
        if self.shuffle_input:
            row_perm = rng.permutation(P.shape[0])
        else:
            row_perm = np.arange(P.shape[0])
        undo_row_perm = np.argsort(row_perm)
        P_perm = P[row_perm]
        if maximize is None:
            maximize = self.maximize
        _, permutation = linear_sum_assignment(P_perm, maximize=maximize)
        return permutation[undo_row_perm]

    def linear_sum_transport(
        self,
        P: np.ndarray,
    ) -> np.ndarray:
        maximize = self.maximize
        reg = self.transport_regularizer

        power = -1 if maximize else 1
        lamb = reg / np.max(np.abs(P))
        ones = np.ones(P.shape[0])
        P_eps, log = sinkhorn(
            ones,
            ones,
            P,
            power / lamb,
            stopThr=self.transport_tol,
            numItermax=self.transport_max_iter,
            log=True,
            warn=False,
        )
        if log["niter"] == self.transport_max_iter - 1:
            warnings.warn(
                "Sinkhorn-Knopp algorithm for solving linear sum transport "
                f"problem did not converge. The final error was {log['err'][-1]} "
                f"and the `transport_tol` was {self.transport_tol}. "
                "You may want to consider increasing "
                "`transport_regularizer`, increasing `transport_max_iter`, or this "
                "could be the result of `transport_tol` set too small."
            )
        return P_eps

    @write_status("Computing step size", 2)
    def compute_step_size(self, P: np.ndarray, Q: np.ndarray) -> float:
        a, b = _compute_coefficients(
            P,
            Q,
            self.A_nn,
            self.B_nn,
            self.AB_nn,
            self.BA_nn,
            self.A_ns,
            self.A_sn,
            self.B_ns,
            self.B_sn,
            self.AB_ns,
            self.AB_sn,
            self.BA_ns,
            self.BA_sn,
            self.S_nn,
        )
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = float(np.argmin([0, (b + a) * self.obj_func_scalar]))
        return alpha

    def check_converged(self, P: np.ndarray, P_new: np.ndarray) -> bool:
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    @write_status("Finalizing assignment", 1)
    def finalize(self, P: np.ndarray, rng: np.random.Generator) -> None:
        self.convex_solution_ = P

        # project back onto the feasible region (permutations)
        if P.shape != (0, 0):
            permutation = self.linear_sum_assignment(P, rng, maximize=True)
        else:  # the case where input was all seeded
            permutation = np.array([], dtype=int)

        # deal with seed-nonseed sorting from the initialization
        permutation = np.concatenate(
            (np.arange(self.n_seeds), permutation + self.n_seeds)
        )
        final_permutation = np.empty(self.n, dtype=int)
        final_permutation[self.perm_A] = self.perm_B[permutation]

        # deal with un-padding
        matching = np.column_stack((np.arange(self.n), final_permutation))
        if self.padded:
            if self._padded_B:
                matching = matching[matching[:, 1] < self.n_B]
            else:
                matching = matching[: self.n_A]

        self.matching_ = matching

        # compute the objective function value for evaluation
        score = self.compute_score(final_permutation)
        self.score_ = score

    def compute_score(self, permutation: np.ndarray) -> float:
        score = 0.0
        n_layers = self.n_layers
        for layer in range(n_layers):
            # casting explicitly to float here because mypy was yelling:
            # 'Incompatible types in assignment (expression has type "floating[Any]",
            # variable has type "float")'
            score += np.sum(self.A[layer] * self.B[layer][permutation][:, permutation])
            score += np.sum(
                self.AB[layer][:, permutation] * self.BA[layer][permutation]
            )
            # for some reason, trace was not working on Py 3.7
            # this is equivalent to trace(SP^T)
            score += float(np.sum(self.S[np.arange(self.S.shape[0]), permutation]))
        return score

    def status(self) -> str:
        if self.n_iter_ > 0:
            return f"[Iteration: {self.n_iter_}]"
        else:
            return "[Pre-loop]"


def _permute_multilayer(
    adjacency: MultilayerAdjacency,
    permutation: np.ndarray,
    rows: bool = True,
    columns: bool = True,
) -> MultilayerAdjacency:
    new_adjacency = []
    for layer_index in range(len(adjacency)):
        layer = adjacency[layer_index]
        if rows:
            layer = layer[permutation]
        if columns:
            layer = layer[:, permutation]
        new_adjacency.append(layer)
    return new_adjacency


def _check_input_matrix(
    A: MultilayerAdjacency, n_layers: Optional[int]
) -> MultilayerAdjacency:
    if isinstance(A, np.ndarray) and (np.ndim(A) == 2):
        A = [A]
    elif isinstance(A, (csr_matrix, csr_array)):
        A = [A]
    elif isinstance(A, list):
        # iterate over to make sure they're all same shape
        first_layer = A[0]
        for i in range(1, len(A)):
            layer = A[i]
            if (layer.shape[0] != first_layer.shape[0]) or (
                layer.shape[1] != first_layer.shape[1]
            ):
                raise ValueError(
                    "Layers in a multilayer network must all share the same shape."
                )
        if isinstance(A[0], np.ndarray):
            A = np.array(A, dtype=float)
        elif isinstance(A[0], csr_matrix):
            pass
    if (n_layers is not None) and (len(A) != n_layers):
        msg = (
            "Input multilayer matrices (A, B, AB, BA) must have the same "
            "number of layers."
        )
        raise ValueError(msg)
    return A


def _compute_gradient(
    P: np.ndarray,
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    AB: MultilayerAdjacency,
    BA: MultilayerAdjacency,
    const_sum: np.ndarray,
) -> np.ndarray:
    n_layers = len(A)
    grad = const_sum.copy()
    for i in range(n_layers):
        grad += (
            A[i] @ P @ B[i].T
            + A[i].T @ P @ B[i]
            + AB[i] @ P.T @ BA[i].T
            + BA[i].T @ P.T @ AB[i]
        )
    return grad


def _compute_coefficients(
    P: np.ndarray,
    Q: np.ndarray,
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    AB: MultilayerAdjacency,
    BA: MultilayerAdjacency,
    A_ns: MultilayerAdjacency,
    A_sn: MultilayerAdjacency,
    B_ns: MultilayerAdjacency,
    B_sn: MultilayerAdjacency,
    AB_ns: MultilayerAdjacency,
    AB_sn: MultilayerAdjacency,
    BA_ns: MultilayerAdjacency,
    BA_sn: MultilayerAdjacency,
    S: AdjacencyMatrix,
) -> Tuple[float, float]:
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # TODO can also refactor to not repeat multiplications like the old code but I was
    # finding it harder to follow that way.
    n_layers = len(A)
    a_cross = 0
    b_cross = 0
    a_intra = 0
    b_intra = 0
    for i in range(n_layers):
        a_cross += np.trace(AB[i].T @ R @ BA[i] @ R)
        b_cross += np.trace(AB[i].T @ R @ BA[i] @ Q) + np.trace(AB[i].T @ Q @ BA[i] @ R)
        b_cross += np.trace(AB_ns[i].T @ R @ BA_ns[i]) + np.trace(
            AB_sn[i].T @ BA_sn[i] @ R
        )
        a_intra += np.trace(A[i] @ R @ B[i].T @ R.T)
        b_intra += np.trace(A[i] @ Q @ B[i].T @ R.T) + np.trace(A[i] @ R @ B[i].T @ Q.T)
        b_intra += np.trace(A_ns[i].T @ R @ B_ns[i]) + np.trace(A_sn[i] @ R @ B_sn[i].T)

    a = a_cross + a_intra
    b = b_cross + b_intra
    b += np.sum(S * R)  # equivalent to S.T @ R

    return a, b


def _split_matrix(
    matrix: AdjacencyMatrix, n: int
) -> Tuple[AdjacencyMatrix, AdjacencyMatrix, AdjacencyMatrix, AdjacencyMatrix]:
    upper, lower = matrix[:n], matrix[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _split_multilayer_matrix(
    matrices: MultilayerAdjacency, n: int
) -> Tuple[
    MultilayerAdjacency, MultilayerAdjacency, MultilayerAdjacency, MultilayerAdjacency
]:
    n_layers = len(matrices)
    seed_to_seed = []
    seed_to_nonseed = []
    nonseed_to_seed = []
    nonseed_to_nonseed = []

    for i in range(n_layers):
        matrix = matrices[i]
        ss, sn, ns, nn = _split_matrix(matrix, n)
        seed_to_seed.append(ss)
        seed_to_nonseed.append(sn)
        nonseed_to_seed.append(ns)
        nonseed_to_nonseed.append(nn)
    return seed_to_seed, seed_to_nonseed, nonseed_to_seed, nonseed_to_nonseed


def _doubly_stochastic(P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for _ in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _multilayer_adj_pad(
    matrices: MultilayerAdjacency, n_padded: int, method: PaddingType
) -> MultilayerAdjacency:
    n1 = matrices[0].shape[0]
    n2 = matrices[0].shape[1]
    if (n1 == n_padded) and (n2 == n_padded):
        return matrices
    else:
        new_matrices: List[AdjacencyMatrix] = []
        for matrix in matrices:
            new_matrices.append(_adj_pad(matrix, n_padded, method))
        return new_matrices


def _adj_pad(
    matrix: AdjacencyMatrix, n_padded: Int, method: PaddingType
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(matrix, (csr_matrix, csr_array)) and (method == "adopted"):
        msg = (
            "Using adopted padding method with a sparse adjacency representation; this "
            "will convert the matrix to a dense representation and likely remove any "
            "speedup from the sparse representation."
        )
        warnings.warn(msg)
        matrix = matrix.toarray()

    if method == "adopted":
        matrix = 2 * matrix - np.ones(matrix.shape)

    if (method == "naive") and isinstance(matrix, (csr_matrix, csr_array)):
        matrix_padded = csr_array((n_padded, n_padded))
    else:
        matrix_padded = np.zeros((n_padded, n_padded))

    matrix_padded[: matrix.shape[0], : matrix.shape[1]] = matrix

    return matrix_padded


def _compare_dimensions(
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    dimension_A: str,
    dimension_B: str,
    name1: str,
    name2: str,
) -> None:
    matrix_A = A[0]
    matrix_B = B[0]
    dim_index_A = 0 if dimension_A == "row" else 1
    dim_index_B = 0 if dimension_B == "row" else 1
    if not (matrix_A.shape[dim_index_A] == matrix_B.shape[dim_index_B]):
        msg = (
            f"Input matrix/matrices `{name1}` number of {dimension_A}s must match "
            f"`{name2}` number of {dimension_B}s."
        )
        raise ValueError(msg)


def _check_partial_match(
    partial_match: Optional[Union[np.ndarray, Tuple]], n1: int, n2: int
) -> np.ndarray:

    _partial_match: np.ndarray
    if partial_match is None:
        _partial_match = np.array([[], []]).T
    elif isinstance(partial_match, tuple):
        _partial_match = np.column_stack(partial_match)
    else:
        _partial_match = np.array(partial_match)

    _partial_match = np.atleast_2d(_partial_match).astype(int)

    _, seeds1_counts = np.unique(_partial_match[:, 0], return_counts=True)
    _, seeds2_counts = np.unique(_partial_match[:, 1], return_counts=True)

    msg = None
    if _partial_match.shape[0] > min(n1, n2):
        msg = "`partial_match` can have only as many seeds as there are nodes"
    elif _partial_match.shape[1] != 2:
        msg = "`partial_match` must have two columns"
    elif _partial_match.ndim != 2:
        msg = "`partial_match` must have exactly two dimensions"
    elif (_partial_match < 0).any():
        msg = "`partial_match` must contain only positive indices"
    elif (_partial_match[:, 0] >= n1).any() or (_partial_match[:, 1] >= n2).any():
        msg = "`partial_match` entries must be less than number of nodes"
    elif (len(_partial_match) > 0) and (
        (seeds1_counts.max() > 1) or (seeds2_counts.max() > 1)
    ):
        msg = "`partial_match` column entries must be unique"

    if msg is not None:
        raise ValueError(msg)

    return _partial_match


def _check_init_input(init: np.ndarray, n: int) -> None:
    if init.ndim != 2:
        msg = "`init` matrix must be 2d"
        raise ValueError(msg)
    row_sum = np.sum(init, axis=0)
    col_sum = np.sum(init, axis=1)
    tol = 1e-3
    msg = ""
    if init.shape != (n, n):
        msg = "`init` matrix must be n x n, where n is the number of non-seeded nodes"
    elif (
        (~np.isclose(row_sum, 1, atol=tol)).any()
        or (~np.isclose(col_sum, 1, atol=tol)).any()
        or (init < 0).any()
    ):
        msg = "`init` matrix must be doubly stochastic"
    if msg != "":
        raise ValueError(msg)
