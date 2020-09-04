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

import ot
import numpy as np
from sklearn.utils import check_array

from .base import BaseAlign
from .sign_flips import SignFlips
from .orthogonal_procrustes import OrthogonalProcrustes


class SeedlessProcrustes(BaseAlign):
    """
    Implements an algorithm that matches two datasets using an orthogonal
    matrix. Unlike OrthogonalProcrustes, this does not use a matching between
    entries. It can even be used in the settings when the two datasets do not
    have the same number of vertices.

    In essence, it aims to simultaneously obtain a, not necessarily 1-to-1,
    correspondance between the vertices of the two data sets, and the
    orthogonal matrix alignment matrix. It does via a sequence of alternating
    steps, similarly to a regular expectation-maximization procedures. The step
    of obtaining an assignment ("E-step") is done by solving an optimal
    transport problem via Sinkhorn algorithm, whereas obtaining an orthogonal
    alignment matrix ("M-step") is done via regular orthogonal procurstes.

    Parameters
    ----------
        optimal_transport_lambda : float, optional
            Regularization term of the Sinkhorn optimal transport algorithm.

        optimal_transport_eps : float, optional
            Tolerance parameter for the each Sinkhorn optimal transport
            algorithm. I.e. tolerance for each "E-step".

        optimal_transport_num_reps : int, optional
            Number of repetitions in each iteration of the iterative optimal
            transport problem. I.e. maximum number of repetitions in each
            "E-step".

        optimal_transport_eps : float, optional
            Tolerance for the each iteration of the iterative optimal transport
            problem. I.e. tolerance of the whole "EM" algorithm.

        iterative_num_reps : int, optional
            Number of reps in each iteration of the iterative optimal transport
            problem. I.e. maxumum number of total iterations the whole "EM"
            algorithm.

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

        freeze_Y : boolean, optional (default True)
            Irrelevant in SeedlessProcrustes, as it always modifies only the
            first dataset. Exists for compatibility with other align modules.

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
        optimal_transport_lambda=0.1,
        optimal_transport_eps=0.01,
        optimal_transport_num_reps=1000,
        iterative_eps=0.01,
        iterative_num_reps=100,
        initialization="2d",
        initial_Q=None,
        initial_P=None,
        freeze_Y=True,
    ):
        # Type checking
        if type(optimal_transport_lambda) is not float:
            msg = "optimal_transport_lambda must be a float, not {}".format(
                type(optimal_transport_lambda)
            )
            raise TypeError(msg)
        if type(optimal_transport_eps) is not float:
            msg = "optimal_transport_eps must be a float, not {}".format(
                type(optimal_transport_eps)
            )
            raise TypeError(msg)
        if type(optimal_transport_num_reps) is not int:
            msg = "optimal_transport_num_reps must be a int, not {}".format(
                type(optimal_transport_num_reps)
            )
            raise TypeError(msg)
        if type(iterative_eps) is not float:
            msg = "iterative_eps must be a float, not {}".format(type(iterative_eps))
            raise TypeError(msg)
        if type(iterative_num_reps) is not int:
            msg = "iterative_num_reps must be a int, not {}".format(
                type(iterative_num_reps)
            )
            raise TypeError(msg)
        if type(initialization) is not str:
            msg = "initalization must be a str, not {}".format(type(initialization))
            raise TypeError(msg)
        if initial_Q is not None:
            if not isinstance(initial_Q, np.ndarray):
                msg = f"initial_Q must be np.ndarray or None, not {type(initial_Q)}"
                raise TypeError(msg)
            initial_Q = check_array(initial_Q, accept_sparse=True, copy=True)
            if initial_Q[0] != initial_Q[1]:
                msg = "initial_Q must be a squared orhthogonal matrix"
                raise ValueError(msg)
            if not (initial_Q.T @ initial_Q == np.eye(initial_Q.shape[0])).all():
                msg = "initial_Q must be a squared orhthogonal matrix"
                raise ValueError(msg)
        if initial_P is not None:
            if not isinstance(initial_P, np.ndarray):
                msg = f"initial_P must be np.ndarray or None, not {type(initial_P)}"
                raise TypeError(msg)
            initial_P = check_array(initial_P, accept_sparse=True, copy=True)

        # Value checking
        if optimal_transport_eps <= 0:
            msg = "{} is an invalud value of the optimal transport eps, must be postitive".format(
                optimal_transport_eps
            )
            raise ValueError(msg)
        if optimal_transport_lambda <= 0:
            msg = "{} is an invalud value of the optimal transport lambda, must be non-negative".format(
                optimal_transport_lambda
            )
            raise ValueError(msg)
        if optimal_transport_num_reps < 1:
            msg = "{} is invalid number of repetitions, must be greater than 1".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        if iterative_eps <= 0:
            msg = (
                "{} is an invalud value of the iterative eps, must be postitive".format(
                    iterative_eps
                )
            )
            raise ValueError(msg)
        if iterative_num_reps < 1:
            msg = "{} is invalid number of repetitions, must be greater than 1".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        initializations_supported = ["2d", "sign_flips", "custom"]
        if initialization not in initializations_supported:
            msg = "supported initializations are {}".format(initialization)
            raise NotImplementedError(msg)


        super().__init__(freeze_Y=freeze_Y)

        self.optimal_transport_eps = optimal_transport_eps
        self.optimal_transport_num_reps = optimal_transport_num_reps
        self.optimal_transport_lambda = optimal_transport_lambda
        self.iterative_eps = iterative_eps
        self.iterative_num_reps = iterative_num_reps
        self.initialization = initialization
        self.initial_Q = initial_Q
        self.initial_P = initial_P

    def _orthogonal_matrix_from_int(self, val_int, d):
        val_bin = bin(val_int)[2:]
        val_bin = "0" * (d - len(val_bin)) + val_bin
        return np.diag(np.array([(float(i) - 0.5) * -2 for i in val_bin]))

    def _optimal_transport(self, X, Y, Q):
        # "E step" of the SeedlessProcrustes.
        n, d = X.shape
        m, _ = Y.shape
        # initialize probability mass arrays & the cost matrix ; run sinkhorn
        probability_mass_X = np.ones(n) / n
        probability_mass_Y = np.ones(m) / m
        cost_matrix = (
            np.linalg.norm((X @ Q).reshape(n, 1, d) - Y.reshape(1, m, d), axis=2) ** 2
        )
        P = ot.sinkhorn(
            a=probability_mass_X,
            b=probability_mass_Y,
            M=cost_matrix,
            reg=self.optimal_transport_lambda,
            numItermax=self.optimal_transport_eps,
            stopThr=self.optimal_transport_eps,
        )
        return P

    def _procrustes(self, X, P_i, Y):
        # "M step" of the SeedlessProcurstes.
        aligner = OrthogonalProcrustes()
        Q = aligner.fit(X, P_i @ Y).Q_X
        return Q

    def _iterative_ot(self, X, Y, Q):
        for i in range(self.iterative_num_reps):
            P_i = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, P_i, Y)
            c = np.linalg.norm(X @ Q - P_i @ Y, ord="fro")
            if c < self.iterative_eps:
                break
        return P_i, Q

    def fit(self, X, Y):
        """
        Uses the two datasets to learn matrices Q_X and Q_Y.
        In seedless procrustes Q_X is a final solution of the to the iterative
        optimal transport / procrustes algorithm and Q_Y is the identity
        matrix.

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

        # check for numpy-ness, 2d-ness and finite-ness
        if not isinstance(X, np.ndarray):
            msg = f"first dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        if not isinstance(Y, np.ndarray):
            msg = f"first dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X, accept_sparse=True, copy=True)
        Y = check_array(Y, accept_sparse=True, copy=True)

        # check for equal components and number of entries
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)
        _, d = X.shape

        if self.initialization == "2d":
            n, d = X.shape
            m, _ = Y.shape
            P_matrices = np.zeros((2 ** d, n, m))
            Q_matrices = np.zeros((2 ** d, d, d))
            objectives = np.zeros(2 ** d)
            # try 2^d different initializations
            for i in range(2 ** d):
                initial_Q = self._orthogonal_matrix_from_int(i, d)
                P_matrices[i], Q_matrices[i] = P, Q = self._iterative_ot(
                    X,
                    Y,
                    initial_Q,
                )
                objectives[i] = np.linalg.norm(X @ Q - P @ Y, ord="fro")
            # pick the best one, using the objective function value
            best = np.argmin(objectives)
            self.initial_Q = self._orthogonal_matrix_from_int(best, d)
            self.P, self.Q_X = P_matrices[best], Q_matrices[best]
        elif self.initialization == "sign_flips":
            aligner = SignFlips(freeze_Y=True)
            self.initial_Q = aligner.fit(X, Y).Q_X
            self.P, self.Q_X = self._iterative_ot(X, Y, initial_Q)
        else:
            # determine initial Q if "custom" and not provided
            if self.initial_Q is None:
                if self.initial_P is not None:
                    # use initial P, if provided
                    self.initial_Q = self._procrustes(X, Y, self.initial_P)
                else:
                    # set to initial Q to identity if neither Q nor P provided
                    self.initial_Q = np.eye(X.shape[1])
            self.P, self.Q_X = self._iterative_ot(X, Y, initial_Q)

        self.Q_Y = np.eye(X.shape[1])

        return self
