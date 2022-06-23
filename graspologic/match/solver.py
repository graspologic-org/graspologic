import time
import warnings
from functools import wraps
from typing import Literal, Optional, Union

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


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def write_status(f, msg, level):
    @wraps(f)
    def wrap(*args, **kw):
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
        transport_implementation: Literal["pot", "ds"] = "pot",
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
        self.transport_implementation = transport_implementation
        self.transport_maxiter = transport_maxiter

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

        if partial_match is None:
            partial_match = np.array([[], []]).astype(int).T
            self._seeded = False
        else:
            self._seeded = True
        self.partial_match = partial_match

        # TODO input validation
        # TODO seeds
        # A, B, partial_match = _common_input_validation(A, B, partial_match)

        # TODO similarity
        # if S is None:
        #     S = np.zeros((A.shape[0], B.shape[1]))
        # S = np.atleast_2d(S)

        # TODO padding

        # TODO make B always bigger

        # convert everything to make sure they are 3D arrays (first dim is layer)
        A = _check_input_matrix(A)
        B = _check_input_matrix(B)

        self.n_A = A[0].shape[0]
        self.n_B = B[0].shape[0]
        self.n_layers = len(A)

        if AB is None:
            AB = np.zeros((self.n_layers, self.n_A, self.n_B))
        if BA is None:
            BA = np.zeros((self.n_layers, self.n_B, self.n_A))

        AB = _check_input_matrix(AB)
        BA = _check_input_matrix(BA)

        self._compute_gradient = _compute_gradient
        self._compute_coefficients = _compute_coefficients
        if isinstance(A[0], csr_matrix):
            self._sparse = True
        else:
            self._sparse = False
            if use_numba:
                self._compute_gradient = _compute_gradient_numba
                self._compute_coefficients = _compute_coefficients_numba

        n_seeds = len(partial_match)
        self.n_seeds = n_seeds
        # set up so that seeds are first and we can grab subgraphs easily
        # TODO could also do this slightly more efficiently just w/ smart indexing?
        nonseed_A = np.setdiff1d(range(A[0].shape[0]), partial_match[:, 0])
        nonseed_B = np.setdiff1d(range(B[0].shape[0]), partial_match[:, 1])
        perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
        perm_B = np.concatenate([partial_match[:, 1], nonseed_B])
        self._undo_perm_A = np.argsort(perm_A)
        self._undo_perm_B = np.argsort(perm_B)

        # permute each (sub)graph appropriately
        A = _permute_multilayer(A, perm_A, rows=True, columns=True)
        B = _permute_multilayer(B, perm_B, rows=True, columns=True)
        AB = _permute_multilayer(AB, perm_A, rows=True, columns=False)
        AB = _permute_multilayer(AB, perm_B, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_A, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_B, rows=True, columns=False)

        # split into subgraphs of seed-to-seed (ss), seed-to-nonseed (sn), etc.
        # main thing being permuted has no subscript
        self.A_ss, self.A_sn, self.A_ns, self.A = _split_matrix(A, n_seeds)
        self.B_ss, self.B_sn, self.B_ns, self.B = _split_matrix(B, n_seeds)
        self.AB_ss, self.AB_sn, self.AB_ns, self.AB = _split_matrix(AB, n_seeds)
        self.BA_ss, self.BA_sn, self.BA_ns, self.BA = _split_matrix(BA, n_seeds)

        self.n_unseed = self.B[0].shape[0]

        if similarity is None:
            similarity = np.zeros((self.n_A, self.n_B))

        similarity = similarity[perm_A][:, perm_B]
        self.S_ss, self.S_sn, self.S_ns, self.S = _split_matrix(
            similarity, n_seeds, single_layer=True
        )

    def solve(self):
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
    def check_outlier_cases(self):
        pass

    @write_status("Initializing", 1)
    def initialize(self):
        if isinstance(self.init, float):
            n_unseed = self.n_unseed
            rng = self.rng
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            # DO linear combo from barycenter
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * self.init + K * (1 - self.init)  # TODO check how defined in paper
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
    def compute_constant_terms(self):
        self.constant_sum = np.zeros((self.n_layers, self.n_B, self.n_B))
        if self._seeded:
            n_layers = len(self.A)
            ipsi = []
            contra = []
            for i in range(n_layers):
                ipsi.append(
                    self.A_ns[i] @ self.B_ns[i].T + self.A_sn[i].T @ self.B_sn[i]
                )
                contra.append(
                    self.AB_ns[i] @ self.BA_ns[i].T + self.BA_sn[i].T @ self.AB_sn[i]
                )
            ipsi = np.array(ipsi)
            contra = np.array(contra)
            self.ipsi_constant_sum = ipsi
            self.contra_constant_sum = contra
            self.constant_sum = ipsi + contra
        self.constant_sum += self.S

    @write_status("Computing gradient", 2)
    def compute_gradient(self, P):
        gradient = self._compute_gradient(
            P, self.A, self.B, self.AB, self.BA, self.constant_sum
        )
        return gradient

    @write_status("Solving assignment problem", 2)
    def compute_step_direction(self, gradient):
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        if self.transport:
            Q = self.linear_sum_transport(gradient)
        else:
            permutation = self.linear_sum_assignment(gradient)
            Q = np.eye(self.n_unseed)[permutation]
        return Q

    def linear_sum_assignment(self, P):
        row_perm = self.rng.permutation(P.shape[1])
        undo_row_perm = np.argsort(row_perm)
        P_perm = P[row_perm]
        _, permutation = linear_sum_assignment(P_perm, maximize=self.maximize)
        return permutation[undo_row_perm]

    def linear_sum_transport(
        self,
        P,
    ):
        maximize = self.maximize
        reg = self.transport_regularizer

        power = -1 if maximize else 1
        lamb = reg / np.max(np.abs(P))
        if self.transport_implementation == "pot":
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
        elif self.transport_implementation == "ds":
            P = np.exp(lamb * power * P)
            P_eps = _doubly_stochastic(
                P, self.transport_tolerance, self.transport_maxiter
            )
        return P_eps

    @write_status("Computing step size", 2)
    def compute_step_size(self, P, Q):
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
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def check_converged(self, P, P_new):
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    @write_status("Finalizing assignment", 1)
    def finalize(self, P):
        self.P_final_ = P

        permutation = self.linear_sum_assignment(P)
        permutation += len(self.partial_match[:, 1])  # TODO this is not robust
        permutation = np.concatenate((self.partial_match[:, 1], permutation))
        self.permutation_ = permutation

        score = self.compute_score(permutation)
        self.score_ = score

    def compute_score(*args):
        return 0

    def status(self):
        if self.n_iter > 0:
            return f"[Iteration: {self.n_iter}]"
        else:
            return "[Pre-loop]"


def _permute_multilayer(adjacency, permutation, rows=True, columns=True):
    for layer_index in range(len(adjacency)):
        layer = adjacency[layer_index]
        if rows:
            layer = layer[permutation]
        if columns:
            layer = layer[:, permutation]
        adjacency[layer_index] = layer
    return adjacency


def _check_input_matrix(A):
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


def _compute_gradient(P, A, B, AB, BA, const_sum):
    n_layers = A.shape[0]
    grad = np.zeros_like(P)
    for i in range(n_layers):
        grad += (
            A[i] @ P @ B[i].T
            + A[i].T @ P @ B[i]
            + AB[i] @ P.T @ BA[i].T
            + BA[i].T @ P.T @ AB[i]
            + const_sum[i]
        )
    return grad


_compute_gradient_numba = njit(_compute_gradient)


def _compute_coefficients(
    P, Q, A, B, AB, BA, A_ns, A_sn, B_ns, B_sn, AB_ns, AB_sn, BA_ns, BA_sn, S
):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares

    n_layers = A.shape[0]
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

# TODO: replace use of this function with sinkhorn in POT
# REF: https://github.com/microsoft/graspologic/blob/dev/graspologic/match/qap.py
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


def _split_matrix(
    matrices: np.ndarray, n: int, single_layer: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if single_layer:
        matrices = [matrices]
    n_layers = len(matrices)
    seed_to_seed = []
    seed_to_nonseed = []
    nonseed_to_seed = []
    nonseed_to_nonseed = []
    for i in range(n_layers):
        X = matrices[i]
        upper, lower = X[:n], X[n:]
        seed_to_seed.append(upper[:, :n])
        seed_to_nonseed.append(upper[:, n:])
        nonseed_to_seed.append(lower[:, :n])
        nonseed_to_nonseed.append(lower[:, n:])
    seed_to_seed = np.array(seed_to_seed)
    seed_to_nonseed = np.array(seed_to_nonseed)
    nonseed_to_seed = np.array(nonseed_to_seed)
    nonseed_to_nonseed = np.array(nonseed_to_nonseed)
    return seed_to_seed, seed_to_nonseed, nonseed_to_seed, nonseed_to_nonseed
