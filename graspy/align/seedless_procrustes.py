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
    have the same number of entries.

    In graph setting, it is used to align the embeddings of two different
    graphs, when it requires some simultaneous inference task, for example,
    inside of the test for the equivalence of the latent distributions, and no
    1-1 matching between the vertices of the two graphs can be established.

    Parameters
    ----------
        optimal_transport_lambda : float (default=0.1), optional
            Regularization term of the Sinkhorn optimal transport algorithm.

        optimal_transport_eps : float (default=0.01), optional
            Tolerance parameter for the each Sinkhorn optimal transport
            algorithm. I.e. tolerance for each "E-step".

        optimal_transport_num_reps : int (default=1000), optional
            Number of repetitions in each iteration of the iterative optimal
            transport problem. I.e. maximum number of repetitions in each
            "E-step".

        iterative_num_reps : int (default=100), optional
            Number of reps in each iteration of the iterative optimal transport
            problem. I.e. maxumum number of total iterations the whole "EM"
            algorithm.

        init : string, {"2d" (default), "sign_flips", "custom"}, optional

            - "2d"
                Uses 2^d different initiazlizations, where d is the dimension.
                specifically, uses all possible matrices with all entries real
                and diagonal entries having magnitude 1 and 0s everywehre else.
                for example, for d=2, tries [[1, 0], [0, 1]], [[1, 0], [0,
                -1]], [[-1, 0], [0, 1]], and [[-1, 0], [0, -1]]. picks the best
                one based on the value of the objective function.
            - 'sign_flips'
                Initial alignment done by making the median value in each
                 dimension have the same sign"
            - "custom"
                Expects either an initial matrix Q or initial matrix P during
                the use of fit or fit_transform (but not both). if neither is
                given initializes to Q = I.

        initial_Q : np.ndarray, shape (d, d) or None, optional (default=None)
            An initial guess for the alignment matrix, Q, if such exists. Only
            one of initial_Q, initial_P can be provided at the same time, and
            only if `init` argument is set to 'custom'. If None, and initial_P
            is also None - initializes `initial_Q` to identity matrix. Must be
            an orthogonal matrix, if provided.

        initial_P : np.ndarray, shape (n, m) or None, optional (default=None)
            Initial guess for the optimal transport matrix, P, if such exists.
            Only one of initial_Q, initial_P can be provided at the same time,
            and only if `init` argument is set to 'custom'. If None, and
            initial_P is also None - initializes `initial_Q` to identity
            matrix. Must be a doubly stochastic matrix if provided (rows sum up
            to 1/n, cols sum up to 1/m.)

    Attributes
    ----------
        Q_ : array, size (d, d)
            Final orthogonal matrix, used to modify X.

        P_ : array, size (n, m) where n and md are the sizes of two datasets
            Final matrix of optimal transports

        score_ : float
            Final value of the objective function: :math:`|| X Q - P Y ||_F`

    References
    ----------
    .. [1] Agterberg, J.

    Notes
    -----
    In essence, the goal of this procedure is to simultaneously obtain a, not
    necessarily 1-to-1, correspondence between the vertices of the two data
    sets, and an orthogonal alignment between two datasets. If the two datasets
    are represented with matrices :math:`X \in M_{n, d}` and :math:`Y \in M_{m,
    d}`, then the correspondence is a matrix :math:`P \in M_{n, m}` that is
    doubly-stochastic (that is, it's rows sum to :math:`1/n`, and columns sum
    to :math:`1/m`) and the orthogonal alignment is an orthogonal matrix
    :math:`Q \in M_{d, d}`. The global objective function is
    :math:`||X W - P Y ||_F`.

    Note that both :math:`X` and :math:`PY` are matrices in :math:`M_{n, d}`.
    Thus, if one knew :math:`P`, it would be simple to obtain an estimate for
    :math:`W`, using the regular orthogonal procrustes. On the other hand, if
    :math:`W` was known, then :math:`XW` and :math:`Y` could be thought of
    distributions over a finite number of masses, each with weight :math:`1/n`
    or :math:`1/m`, respectively. These distributions could be "matched" via
    solving an optimal transport problem.

    However, both :math:`W` and :math:`P` are simultaneously unknown here. So
    the algorithm performs a sequence of alternating steps, obtaining
    iteratively improving estimates of :math:`W` and :math:`P`, similarly to an
    expectation-maximization (EM) procedure. It is not known whether this
    procedure is formally an EM, but the analogy can be drawn as follows: after
    obtaining an initial guess of of :math:`\hat{W}_{0}`, obtaining an
    assignment matrix :math:`\hat{P}_{i+1} | \hat{W}_{i}` ("E-step") is done by
    solving an optimal transport problem via Sinkhorn algorithm, whereas
    obtaining an orthogonal alignment matrix :math:`W_{i+1} | P_{i}` ("M-step")
    is done via regular orthogonal procurstes. These alternating steps are
    performed until `iterative_num_reps` is reached.

    For more on how the initial guess can be performed, see `init`.

    """

    def __init__(
        self,
        optimal_transport_lambda=0.1,
        optimal_transport_eps=0.01,
        optimal_transport_num_reps=1000,
        iterative_num_reps=100,
        init="2d",
        initial_Q=None,
        initial_P=None,
    ):
        # check optimal_transport_lambda argument
        if type(optimal_transport_lambda) is not float:
            msg = "optimal_transport_lambda must be a float, not {}".format(
                type(optimal_transport_lambda)
            )
            raise TypeError(msg)
        if optimal_transport_lambda < 0:
            msg = "{} is an invalud value of the optimal_transport_lambda, must be non-negative".format(
                optimal_transport_lambda
            )
            raise ValueError(msg)
        # check optimal_transport_lambda argument
        if type(optimal_transport_eps) is not float:
            msg = "optimal_transport_eps must be a float, not {}".format(
                type(optimal_transport_eps)
            )
            raise TypeError(msg)
        if optimal_transport_eps <= 0:
            msg = "{} is an invalid value of the optimal transport eps, must be postitive".format(
                optimal_transport_eps
            )
            raise ValueError(msg)
        # check optimal_transport_num_reps argument
        if type(optimal_transport_num_reps) is not int:
            msg = "optimal_transport_num_reps must be a int, not {}".format(
                type(optimal_transport_num_reps)
            )
            raise TypeError(msg)
        if optimal_transport_num_reps < 1:
            msg = "{} is an invalid number of repetitions, must be non-negative".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check iterative_num_reps argument
        if type(iterative_num_reps) is not int:
            msg = "iterative_num_reps must be a int, not {}".format(
                type(iterative_num_reps)
            )
            raise TypeError(msg)
        if iterative_num_reps < 1:
            msg = "{} is an invalid number of repetitions, must be non-negative".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check init argument
        if type(init) is not str:
            msg = "initalization must be a str, not {}".format(type(init))
            raise TypeError(msg)
        inits_supported = ["2d", "sign_flips", "custom"]
        if init not in inits_supported:
            msg = "supported inits are {}".format(inits_supported)
            raise ValueError(msg)
        # check that initial_Q and intial_P aren't provided when shouldn't be
        if initial_Q is not None and init != "custom":
            msg = "initial_Q can only be provided if init is set to custom"
            raise ValueError(msg)
        if initial_P is not None and init != "custom":
            msg = "initial_P can only be provided if init is set to custom"
            raise ValueError(msg)
        if initial_Q is not None and initial_P is not None:
            msg = "initial_Q and initial_P cannot be provided simultaneously"
            raise ValueError(msg)
        # check initial_Q argument
        if initial_Q is not None:
            if not isinstance(initial_Q, np.ndarray):
                msg = f"initial_Q must be np.ndarray or None, not {type(initial_Q)}"
                raise TypeError(msg)
            initial_Q = check_array(initial_Q, copy=True)
            if initial_Q.shape[0] != initial_Q.shape[1]:
                msg = "initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
            if not np.allclose(initial_Q.T @ initial_Q, np.eye(initial_Q.shape[0])):
                msg = "initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
        # check initial_P argument
        if initial_P is not None:
            if not isinstance(initial_P, np.ndarray):
                msg = f"initial_P must be np.ndarray or None, not {type(initial_P)}"
                raise TypeError(msg)
            initial_P = check_array(initial_P, copy=True)
            n, m = initial_P.shape
            if not (
                np.allclose(initial_P.sum(axis=0), np.ones(m) / m)
                and np.allclose(initial_P.sum(axis=1), np.ones(n) / n)
            ):
                msg = (
                    "initial_P must be a doubly stochastic matrix "
                    "(rows add up to (1/number of cols) "
                    "and columns add up to (1/number of rows))"
                )
                raise ValueError(msg)

        super().__init__()

        self.optimal_transport_eps = optimal_transport_eps
        self.optimal_transport_num_reps = optimal_transport_num_reps
        self.optimal_transport_lambda = optimal_transport_lambda
        self.iterative_num_reps = iterative_num_reps
        self.init = init
        self.initial_Q = initial_Q
        self.initial_P = initial_P

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

    def _procrustes(self, X, Y, P):
        # "M step" of the SeedlessProcurstes.
        aligner = OrthogonalProcrustes()
        Q = aligner.fit(X, P @ Y).Q_
        return Q

    def _iterative_ot(self, X, Y, Q):
        for i in range(self.iterative_num_reps):
            P = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, Y, P)
            c = self._compute_objective(X, Y, Q, P)
        return P, Q

    def _compute_objective(self, X, Y, Q=None, P=None):
        if Q is None:
            Q = self.Q_
        if P is None:
            P = self.P_
        return np.linalg.norm(X @ Q - P @ Y, ord="fro")

    def fit(self, X, Y):
        """
        Uses the two datasets to learn the matrix Q_ that aligns the first
        dataset with the second.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.

        Returns
        -------
        self : returns an instance of self
        """
        X, Y = self._check_datasets(X, Y)
        n, d = X.shape
        m, _ = Y.shape

        if self.init == "2d":
            P_matrices = np.zeros((2 ** d, n, m))
            Q_matrices = np.zeros((2 ** d, d, d))
            objectives = np.zeros(2 ** d)
            # try 2^d different initializations
            for i in range(2 ** d):
                initial_Q = _sign_flip_matrix_from_int(i, d)
                P_matrices[i], Q_matrices[i] = P, Q = self._iterative_ot(
                    X, Y, initial_Q
                )
                objectives[i] = self._compute_objective(X, Y, Q, P)
            # pick the best one, using the objective function value
            best = np.argmin(objectives)
            self.initial_Q = _sign_flip_matrix_from_int(best, d)
            self.P_, self.Q_ = P_matrices[best], Q_matrices[best]
        elif self.init == "sign_flips":
            aligner = SignFlips()
            self.initial_Q = aligner.fit(X, Y).Q_
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.initial_Q)
        else:
            # determine initial Q if "custom"
            if self.initial_Q is not None:
                pass
            elif self.initial_P is not None:
                # use initial P, if provided
                self.initial_Q = self._procrustes(X, Y, self.initial_P)
            else:
                # set to initial Q to identity if neither Q nor P provided
                self.initial_Q = np.eye(d)
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.initial_Q)
        self.score_ = self._compute_objective(X, Y)

        return self


def _sign_flip_matrix_from_int(val_int, d):
    val_bin = bin(val_int)[2:]
    val_bin = "0" * (d - len(val_bin)) + val_bin
    return np.diag(np.array([(float(i) - 0.5) * -2 for i in val_bin]))
