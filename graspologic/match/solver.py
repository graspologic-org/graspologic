import time
import warnings
from functools import wraps
from typing import Callable, Literal, Optional, Union

import numpy as np
from beartype import beartype
from numba import njit
from ot import sinkhorn
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from graspologic.types import AdjacencyMatrix, List, Tuple

# Type aliases
PaddingType = Literal["adopted", "naive"]
InitMethodType = Literal["barycenter", "rand", "randomized"]
RandomStateType = Optional[Union[int, np.random.RandomState, np.random.Generator]]
ArrayLikeOfIndexes = Union[List[int], np.ndarray]
MultilayerAdjacency = Union[List[AdjacencyMatrix], AdjacencyMatrix, np.ndarray]
Scalar = Union[int, float, np.integer]
Int = Union[int, np.integer]


def parametrized(dec: Callable) -> Callable:
    def layer(*args, **kwargs) -> Callable:  # type: ignore
        def repl(f: Callable) -> Callable:
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
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


class GraphMatchSolver(BaseEstimator):
    """GraphMatchSolver

    This is a draft implementation of a class for solving the graph matching/quadratic
    assignment problems and various extensions thereof. This solver supports:
    - the original FAQ algorithm
    - seeded matching (though this is a work in progress and may only work in some cases)
    - multilayer matching, a generalization where each graph is a multigraph
    - GOAT, which is a modification of the original FAQ algorithm
    - bisected graph matching, allowing for connections between the graphs
    - matching of sparse matrices
    - matching using numba compilation for dense matrices
    - a similarity term between networks

    This class will ultimately be cleaned up and included in graspologic, likely with
    a functional wrapper.

    """

    @beartype
    def __init__(
        self,
        A: MultilayerAdjacency,
        B: MultilayerAdjacency,
        AB: Optional[MultilayerAdjacency] = None,
        BA: Optional[MultilayerAdjacency] = None,
        similarity: Optional[AdjacencyMatrix] = None,
        partial_match: Optional[np.ndarray] = None,
        rng: Optional[RandomStateType] = None,
        init: Optional[Scalar] = 1.0,
        verbose: Int = False,  # 0 is nothing, 1 is loops, 2 is loops + sub, 3, is loops + sub + timing
        shuffle_input: bool = True,
        maximize: bool = True,
        maxiter: Int = 30,
        tol: Scalar = 0.01,
        transport: bool = False,
        use_numba: bool = True,
        transport_regularizer: Scalar = 100,
        transport_tolerance: Scalar = 5e-2,
        transport_maxiter: Int = 1000,
    ):
        # TODO more input checking
        self.rng = check_random_state(rng)
        self.init = init
        self.verbose = verbose
        self.shuffle_input = shuffle_input
        self.maximize = maximize
        self.maxiter = maxiter
        self.tol = tol

        self.transport = transport
        self.transport_regularizer = transport_regularizer
        self.transport_tolerance = transport_tolerance
        self.transport_maxiter = transport_maxiter

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

        if partial_match is None:
            seeds = np.array([[], []]).astype(int).T
            self._seeded = False
        else:
            self._seeded = True
            seeds = partial_match

        # TODO input validation
        # TODO seeds
        # A, B, seeds = _common_input_validation(A, B, seeds)

        # convert everything to make sure they are 3D arrays (first dim is layer)
        A = _check_input_matrix(A)
        B = _check_input_matrix(B)

        # get some useful sizes
        self.n_A = A[0].shape[0]
        self.n_B = B[0].shape[0]
        self.n_layers = len(A)
        n_seeds = len(seeds)
        self.n_seeds = n_seeds

        # check for between-graph terms
        if AB is None:
            AB = np.zeros((self.n_layers, self.n_A, self.n_B))
        else:
            AB = _check_input_matrix(AB)
        if BA is None:
            BA = np.zeros((self.n_layers, self.n_B, self.n_A))
        else:
            BA = _check_input_matrix(BA)

        # check for similarity term
        if similarity is None:
            similarity = np.zeros((self.n_A, self.n_B))

        # TODO padding here
        # TODO make B always bigger

        # set up so that seeds are first and we can grab subgraphs easily
        # TODO could also do this slightly more efficiently just w/ smart indexing?
        # TODO I think this is kind making the assumption that the input seeds
        # is sorted
        sort_inds = np.argsort(seeds[:, 0])
        seeds = seeds[sort_inds]
        nonseed_A = np.setdiff1d(np.arange(A[0].shape[0]), seeds[:, 0])
        nonseed_B = np.setdiff1d(range(B[0].shape[0]), seeds[:, 1])
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
        S = similarity[perm_A][:, perm_B]

        # split into subgraphs of seed-to-seed (ss), seed-to-nonseed (sn), etc.
        # main thing being permuted has no subscript
        self.A_ss, self.A_sn, self.A_ns, self.A = _split_multilayer_matrix(A, n_seeds)
        self.B_ss, self.B_sn, self.B_ns, self.B = _split_multilayer_matrix(B, n_seeds)
        self.AB_ss, self.AB_sn, self.AB_ns, self.AB = _split_multilayer_matrix(
            AB, n_seeds
        )
        self.BA_ss, self.BA_sn, self.BA_ns, self.BA = _split_multilayer_matrix(
            BA, n_seeds
        )

        self.n_unseed = self.B[0].shape[0]

        self.S_ss, self.S_sn, self.S_ns, self.S = _split_matrix(S, n_seeds)

        # decide whether to use numba/sparse
        self._compute_gradient = _compute_gradient
        self._compute_coefficients = _compute_coefficients
        # TODO probably should base this on "ALL" instead of first one being sparse
        if isinstance(A[0], csr_matrix):
            self._sparse = True
        else:
            self._sparse = False
            if use_numba:
                self._compute_gradient = _compute_gradient_numba
                self._compute_coefficients = _compute_coefficients_numba

    def solve(self) -> None:
        self.n_iter = 0
        self.check_outlier_cases()

        P = self.initialize()
        self.compute_constant_terms()
        for n_iter in range(self.maxiter):
            self.n_iter = n_iter + 1

            gradient = self.compute_gradient(P)
            Q = self.compute_step_direction(gradient)
            alpha = self.compute_step_size(P, Q)

            # take a step in this direction
            P_new = alpha * P + (1 - alpha) * Q

            if self.check_converged(P, P_new):
                self.converged = True
                P = P_new
                break
            P = P_new

        self.finalize(P)

    # TODO
    def check_outlier_cases(self) -> None:
        pass

    @write_status("Initializing", 1)
    def initialize(self) -> np.ndarray:
        if isinstance(self.init, float):
            n_unseed = self.n_unseed
            rng = self.rng
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            if self.init < 1:
                # DO linear combo from barycenter
                K = rng.uniform(size=(n_unseed, n_unseed))
                # Sinkhorn balancing
                K = _doubly_stochastic(K)
                P = J * self.init + K * (
                    1 - self.init
                )  # TODO check how defined in paper
            else:
                P = J
        elif isinstance(self.init, np.ndarray):
            raise NotImplementedError()
            # TODO fix below
            # P0 = np.atleast_2d(P0)
            # _check_init_input(P0, n_unseed)
            # invert_inds = np.argsort(nonseed_B)
            # perm_nonseed_B = np.argsort(invert_inds)
            # P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    @write_status("Computing constant terms", 2)
    def compute_constant_terms(self) -> None:
        self.constant_sum = np.zeros((self.n_unseed, self.n_unseed))
        if self._seeded:
            n_layers = len(self.A)
            for i in range(n_layers):
                self.constant_sum += (
                    self.A_ns[i] @ self.B_ns[i].T  # ipsi
                    + self.A_sn[i].T @ self.B_sn[i]  # ipsi
                    + self.AB_ns[i] @ self.BA_ns[i].T  # contra
                    + self.BA_sn[i].T @ self.AB_sn[i]  # contra
                )
        self.constant_sum += self.S

        # self.constant_sum = constant_sum
        # contra.append()
        #     self.ipsi_constant_sum = np.array(ipsi)
        #     self.contra_constant_sum = np.array(contra)
        #     self.constant_sum = self.ipsi_constant_sum + self.contra_constant_sum
        # print("constant_sum")
        # print(type(self.constant_sum))
        # print("S")
        # print(type(self.S))
        # self.constant_sum = np.array(self.constant_sum) + np.array(self.S)

    @write_status("Computing gradient", 2)
    def compute_gradient(self, P: np.ndarray) -> np.ndarray:
        gradient = self._compute_gradient(
            P, self.A, self.B, self.AB, self.BA, self.constant_sum
        )
        return gradient

    @write_status("Solving assignment problem", 2)
    def compute_step_direction(self, gradient: np.ndarray) -> np.ndarray:
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        if self.transport:
            Q = self.linear_sum_transport(gradient)
        else:
            permutation = self.linear_sum_assignment(gradient)
            Q = np.eye(self.n_unseed)[permutation]
        return Q

    def linear_sum_assignment(self, P: np.ndarray) -> np.ndarray:
        """This is a modified version of LAP which (in expectation) does not care
        about the order of the inputs. This matters because scipy LAP settles ties
        (which do come up) based on the ordering of the inputs. This can lead to
        artificially high matching accuracy when the user inputs data which is in the
        correct permutation, for example."""
        row_perm = self.rng.permutation(P.shape[1])
        undo_row_perm = np.argsort(row_perm)
        P_perm = P[row_perm]
        _, permutation = linear_sum_assignment(P_perm, maximize=self.maximize)
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
            stopThr=self.transport_tolerance,
            numItermax=self.transport_maxiter,
            log=True,
            warn=False,
        )
        if log["niter"] == self.transport_maxiter - 1:
            warnings.warn(
                "Sinkhorn-Knopp algorithm for solving linear sum transport "
                f"problem did not converge. The final error was {log['err'][-1]} "
                f"and the `transport_tolerance` was {self.transport_tolerance}. "
                "You may want to consider increasing "
                "`transport_regularizer`, increasing `transport_maxiter`, or this "
                "could be the result of `transport_tolerance` set too small."
            )
        return P_eps

    @write_status("Computing step size", 2)
    def compute_step_size(self, P: np.ndarray, Q: np.ndarray) -> float:
        a, b = self._compute_coefficients(
            P,
            Q,
            self.A,
            self.B,
            self.AB,
            self.BA,
            self.A_ns,
            self.A_sn,
            self.B_ns,
            self.B_sn,
            self.AB_ns,
            self.AB_sn,
            self.BA_ns,
            self.BA_sn,
            self.S,
        )
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = float(np.argmin([0, (b + a) * self.obj_func_scalar]))
        return alpha

    def check_converged(self, P: np.ndarray, P_new: np.ndarray) -> bool:
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    @write_status("Finalizing assignment", 1)
    def finalize(self, P: np.ndarray) -> None:
        self.P_final_ = P

        permutation = self.linear_sum_assignment(P)
        permutation = np.concatenate(
            (np.arange(self.n_seeds), permutation + self.n_seeds)
        )
        # not_seeds = np.setdiff1d(np.arange(self.n_B), self.seeds[:, 1])

        final_permutation = np.empty(self.n_B, dtype=int)
        final_permutation[self.perm_A] = self.perm_B[permutation]
        # final_permutation[self.seeds[:, 0]] = self.seeds[:, 1]
        # final_permutation[not_seeds] = permutation
        # permutation += len(self.seeds[:, 1])  # TODO this is not robust
        # permutation = np.concatenate((self.seeds[:, 1], not_seeds[permutation]))
        self.permutation_ = final_permutation

        score = self.compute_score(permutation)
        self.score_ = score

    def compute_score(self, permutation: np.ndarray) -> float:
        return 0.0

    def status(self) -> str:
        if self.n_iter > 0:
            return f"[Iteration: {self.n_iter}]"
        else:
            return "[Pre-loop]"


def _permute_multilayer(
    adjacency: MultilayerAdjacency,
    permutation: np.ndarray,
    rows: bool = True,
    columns: bool = True,
) -> MultilayerAdjacency:
    for layer_index in range(len(adjacency)):
        layer = adjacency[layer_index]
        if rows:
            layer = layer[permutation]
        if columns:
            layer = layer[:, permutation]
        adjacency[layer_index] = layer
    return adjacency


def _check_input_matrix(A: MultilayerAdjacency) -> MultilayerAdjacency:
    if isinstance(A, np.ndarray) and (np.ndim(A) == 2):
        A = np.expand_dims(A, axis=0)
        A = A.astype(float)
    elif isinstance(A, csr_matrix):
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
    return A


def _compute_gradient(
    P: np.ndarray,
    A: MultilayerAdjacency,
    B: MultilayerAdjacency,
    AB: MultilayerAdjacency,
    BA: MultilayerAdjacency,
    const_sum: MultilayerAdjacency,
) -> np.ndarray:
    n_layers = len(A)
    grad = const_sum
    for i in range(n_layers):
        grad += (
            A[i] @ P @ B[i].T
            + A[i].T @ P @ B[i]
            + AB[i] @ P.T @ BA[i].T
            + BA[i].T @ P.T @ AB[i]
        )

    return grad


_compute_gradient_numba = njit(_compute_gradient)


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
    # though I don't know how much Numba cares

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


_compute_coefficients_numba = njit(_compute_coefficients)


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

    # if isinstance(X, np.ndarray):
    # _seed_to_seed = np.array(seed_to_seed)
    # _seed_to_nonseed = np.array(seed_to_nonseed)
    # _nonseed_to_seed = np.array(nonseed_to_seed)
    # _nonseed_to_nonseed = np.array(nonseed_to_nonseed)
    # return _seed_to_seed, _seed_to_nonseed, _nonseed_to_seed, _nonseed_to_nonseed


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
