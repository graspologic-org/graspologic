# Copyright 2020 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils import check_array

from .base import BaseAlign


class SeedlessProcrustes(BaseAlign):
    """
    Matches datasets by iterating over a decreasing sequence of lambdas,
    the penalization parameters.
    If lambda is big, the function is more concave, so we iterate, starting
    from Lambda = .5, and decreasing each time by alpha.
    This method takes longer, but is more likely to converge to the true
    solution, since we start from a more concave problem and iteratively solve
    it by setting lambda = alpha * lambda, for alpha in (0,1).

    Parameters
    ----------
        freeze_Y : boolean, optional (default True)
            Irrelevant in SeedlessProcrustes, as it always modifies only one
            dataset. Exists for compatibility with other align modules.

        lambda_init : float, optional (default 0.5)
            Initial value of the lambda penalization parameter in the overall
            aligning problem.

        lambda_final : float, optional (default: 0.001)
            Termination value of the lambda penalization parameters in the
            overall aligning problem.

        alpha : float, optional (default 0.95)
            Parameter by which lambda is multiplied by in every step in the
            overall aligning problem.

        num_reps : int, optional
            Number of reps in each iteration of the optimal transport problem.

        iteration_eps : float, optional
            Tolerance for the each iteration of the optimal transport problem.

        optimal_transport_eps : float, optional
            Tolerance parameter for the each subiteration of the optimal
            transport problem.

        initialization: string, {"2d", "sign_flips", "custom"}

            - "2d"
                uses 2^d different initiazlizations, where d is the dimension.
                specifically, uses all possible matrices with all entries real
                and diagonal entries having magnitude 1 and 0s everywehre else.
                for example, for d=2, tries [[1, 0], [0, 1]], [[1, 0], [0,
                -1]], [[-1, 0], [0, 1]], and [[-1, 0], [0, -1]]. picks the best
                one based on the value of the objective function.
            - 'sign_flips'
                for the first dimension, if two datasets have medians with
                varying signs, flips all signs along this dimension for the
                first dataset. then initializes to an identity.
            - "custom"
                expects either an initial matrix Q or initial matrix P during
                the use of fit or fit_transform. uses initial Q provided,
                unless it is not provided. if not provided - uses initial P. if
                neither is given initializes to Q = I.

        initial_Q: np.ndarray, shape (d, d) or None, optional (default=None)
            An initial guess for the alignment matrix, if such exists. Ignored
            if initialization alignment is set to anything other than 'custom'.
            If None - initializes using an initial guess for P.
            If None, and P is also None - initializes Q to identity matrix.

        initial_P: np.ndarray, shape (n, m) or None, optional (default=None)
            Initial guess for the initial transport matrix.
            Only matters if Q=None.

    Attributes
    ----------
        Q_X : array, size (d, d)
              final orthogonal matrix, used to modify X

        Q_Y : array, size (d, d)
              final orthogonal matrix, used to modify Y.
              in SeedlessProcrustes Q_Y is always equal to identity I

        P : array, size (n, m) where n and md are the sizes of two datasets
            final matrix of optimal transports

    References
    ----------
    .. [1] Agterberg, J.
    """

    def __init__(
        self,
        freeze_Y=True,
        lambda_init=0.5,
        lambda_final=0.001,
        alpha=0.95,
        optimal_transport_eps=0.01,
        iterative_eps=0.01,
        num_reps=100,
        initialization="2d",
        initial_Q=None,
        initial_P=None,
    ):
        if type(lambda_init) is not float:
            raise TypeError()
        if type(lambda_final) is not float:
            raise TypeError()
        if type(alpha) is not float:
            raise TypeError()
        if type(optimal_transport_eps) is not float:
            raise TypeError()
        if type(iterative_eps) is not float:
            raise TypeError()
        if type(num_reps) is not int:
            raise TypeError()
        if alpha < 0 or alpha > 1:
            msg = "{} is an invalid value of alpha must be strictly between 0 and 1".format(
                alpha
            )
            raise ValueError(msg)
        if optimal_transport_eps <= 0:
            msg = "{} is an invalud value of the optimal transport eps, must be postitive".format(
                optimal_transport_eps
            )
            raise ValueError(msg)
        if iterative_eps <= 0:
            msg = (
                "{} is an invalud value of the iterative eps, must be postitive".format(
                    iterative_eps
                )
            )
            raise ValueError(msg)
        if num_reps < 1:
            msg = "{} is invalid number of repetitions, must be greater than 1".format(
                num_reps
            )
            raise ValueError(msg)

        if not isinstance(initialization, str):
            msg = "initialization must be None or a str, not {}".format(
                type(initialization)
            )
            raise TypeError(msg)
        else:
            initializations_supported = ["2d", "sign_flips", "custom"]
            if initialization not in initializations_supported:
                msg = "supported initializations are {}".format(initialization)
                raise NotImplementedError(msg)

        if initial_Q is not None:
            initial_Q = check_array(initial_Q, accept_sparse=True, copy=True)
        if initial_P is not None:
            initial_P = check_array(initial_P, accept_sparse=True, copy=True)

        super().__init__(freeze_Y=freeze_Y)

        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.alpha = alpha
        self.optimal_transport_eps = optimal_transport_eps
        self.iterative_eps = iterative_eps
        self.num_reps = num_reps
        self.initialization = initialization
        self.initial_Q = initial_Q
        self.initial_P = initial_P

    def _procrustes(self, X, Y, P_i):
        u, w, vt = np.linalg.svd(X.T @ P_i @ Y)
        Q = u.dot(vt)
        return Q

    def _optimal_transport(self, X, Y, Q, lambd=0.1):
        n, d = X.shape
        m, _ = Y.shape

        X = X @ Q
        C = np.linalg.norm(X.reshape(n, 1, d) - Y.reshape(1, m, d), axis=2) ** 2

        r = 1 / n
        c = 1 / m
        P = np.exp(-lambd * C)
        u = np.zeros(n)
        while np.max(np.abs(u - np.sum(P, axis=1))) > self.optimal_transport_eps:
            u = np.sum(P, axis=1)
            P = r * P / u.reshape(-1, 1)
            v = np.sum(P, axis=0)
            P = c * (P.T / v.reshape(-1, 1)).T
        return P

    def _iterative_optimal_transport(self, X, Y, Q=None, lambd=0.01):
        _, d = X.shape
        if Q is None:
            Q = np.eye(d)

        for i in range(self.num_reps):
            P_i = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, Y, P_i)
            c = np.linalg.norm(X @ Q - P_i @ Y, ord="fro")
            if c < self.iterative_eps:
                break
        return P_i, Q

    def _sign_flips(self, X, Y):
        X_medians = np.median(X, axis=0)
        Y_medians = np.median(Y, axis=0)
        val = np.multiply(X_medians, Y_medians)
        t = (val > 0) * 2 - 1
        return np.diag(t)

    def _orthogonal_matrix_from_int(self, val_int, d):
        val_bin = bin(val_int)[2:]
        val_bin = "0" * (d - len(val_bin)) + val_bin
        return np.diag(np.array([(float(i) - 0.5) * -2 for i in val_bin]))

    def _match_datasets(self, X, Y, Q):
        lambda_current = self.lambda_init
        while lambda_current > self.lambda_final:
            P_i, Q = self._iterative_optimal_transport(X, Y, Q, lambd=lambda_current)
            lambda_current = self.alpha * lambda_current
        return P_i, Q

    def fit(self, X, Y):
        """
        Matches two datasets using the seedless procrustes algorithm

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y: np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.

        Returns
        -------
        self: returns an instance of self
        """
        # perform checks
        X = check_array(X, accept_sparse=True, copy=True)
        Y = check_array(Y, accept_sparse=True, copy=True)
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)

        if self.initialization == "2d":
            n, d = X.shape
            m, _ = Y.shape
            P_matrices = np.zeros((2 ** d, n, m))
            Q_matrices = np.zeros((2 ** d, d, d))
            objectives = np.zeros(2 ** d)
            # try 2^d different initializations
            for i in range(2 ** d):
                initial_Q = self._orthogonal_matrix_from_int(i, d)
                P_matrices[i], Q_matrices[i] = P, Q = self._match_datasets(
                    X, Y, initial_Q
                )
                objectives[i] = np.linalg.norm(X @ Q - P @ Y, ord="fro")
            # pick the best one, using the objective function value
            best = np.argmin(objectives)
            self.initial_Q = self._orthogonal_matrix_from_int(best, d)
            self.P, self.Q_X = P_matrices[best], Q_matrices[best]
        elif self.initialization == "sign_flips":
            self.initial_Q = self._sign_flips(X, Y)
            self.P, self.Q_X = self._match_datasets(X, Y, self.initial_Q)
        else:
            # determine initial Q, if custom and not provided
            if self.initial_Q is None:
                if self.initial_P is not None:
                    # use initial P, if provided
                    self.initial_Q = self._procrustes(X, Y, self.initial_P)
                else:
                    # set to initial Q to identity if neither Q nor P provided
                    self.initial_Q = np.eye(X.shape[1])
            self.P, self.Q_X = self._match_datasets(X, Y, self.initial_Q)

        self.Q_Y = np.eye(X.shape[1])

        return self
