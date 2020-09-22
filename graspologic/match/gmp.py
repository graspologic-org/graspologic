# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
from .skp import SinkhornKnopp


class GraphMatch(BaseEstimator):
    """
    This class solves the Graph Matching Problem and the Quadratic Assignment Problem
    (QAP) through an implementation of the Fast Approximate QAP Algorithm (FAQ) (these
    two problems are the same up to a sign change) [1].

    This algorithm can be thought of as finding an alignment of the vertices of two
    graphs which minimizes the number of induced edge disagreements, or, in the case
    of weighted graphs, the sum of squared differences of edge weight disagreements.
    The option to add seeds (known vertex correspondence between some nodes) is also
    available [2].


    Parameters
    ----------

    n_init : int, positive (default = 1)
        Number of random initializations of the starting permutation matrix that
        the FAQ algorithm will undergo. n_init automatically set to 1 if
        init_method = 'barycenter'

    init_method : string (default = 'barycenter')
        The initial position chosen

        "barycenter" : the non-informative “flat doubly stochastic matrix,”
        :math:`J=1*1^T /n` , i.e the barycenter of the feasible region

        "rand" : some random point near :math:`J, (J+K)/2`, where K is some random doubly
        stochastic matrix

    max_iter : int, positive (default = 30)
        Integer specifying the max number of Franke-Wolfe iterations.
        FAQ typically converges with modest number of iterations.

    shuffle_input : bool (default = True)
        Gives users the option to shuffle the nodes of A matrix to avoid results
        from inputs that were already matched.

    eps : float (default = 0.1)
        A positive, threshold stopping criteria such that FW continues to iterate
        while Frobenius norm of :math:`(P_{i}-P_{i+1}) > eps`

    gmp : bool (default = True)
        Gives users the option to solve QAP rather than the Graph Matching Problem
        (GMP). This is accomplished through trivial negation of the objective function.

    Attributes
    ----------

    perm_inds_ : array, size (n,) where n is the number of vertices in the fitted graphs.
        The indices of the optimal permutation (with the fixed seeds given) on the nodes of B,
        to best minimize the objective function :math:`f(P) = trace(A^T PBP^T )`.


    score_ : float
        The objective function value of for the optimal permutation found.


    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik, S.G. Kratzer,
        E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and C.E. Priebe, “Fast
        approximate quadratic programming for graph matching,” PLOS one, vol. 10,
        no. 4, p. e0121002, 2015.

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski, C. Priebe,
        Seeded graph matching, Pattern Recognit. 87 (2019) 203–215



    """

    def __init__(
        self,
        n_init=1,
        init_method="barycenter",
        max_iter=30,
        shuffle_input=True,
        eps=0.1,
        gmp=True,
    ):

        if type(n_init) is int and n_init > 0:
            self.n_init = n_init
        else:
            msg = '"n_init" must be a positive integer'
            raise TypeError(msg)
        if init_method == "rand":
            self.init_method = "rand"
        elif init_method == "barycenter":
            self.init_method = "barycenter"
            self.n_init = 1
        else:
            msg = 'Invalid "init_method" parameter string'
            raise ValueError(msg)
        if max_iter > 0 and type(max_iter) is int:
            self.max_iter = max_iter
        else:
            msg = '"max_iter" must be a positive integer'
            raise TypeError(msg)
        if type(shuffle_input) is bool:
            self.shuffle_input = shuffle_input
        else:
            msg = '"shuffle_input" must be a boolean'
            raise TypeError(msg)
        if eps > 0 and type(eps) is float:
            self.eps = eps
        else:
            msg = '"eps" must be a positive float'
            raise TypeError(msg)
        if type(gmp) is bool:
            self.gmp = gmp
        else:
            msg = '"gmp" must be a boolean'
            raise TypeError(msg)

    def fit(self, A, B, seeds_A=[], seeds_B=[]):
        """
        Fits the model with two assigned adjacency matrices

        Parameters
        ----------
        A : 2d-array, square, positive
            A square adjacency matrix

        B : 2d-array, square, positive
            A square adjacency matrix

        seeds_A : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `A`.

        seeds_B : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `B` The elements of
            `seeds_A` and `seeds_B` are vertices which are known to be matched, that is,
            `seeds_A[i]` is matched to vertex `seeds_B[i]`.

        Returns
        -------
        self : returns an instance of self
        """
        A = check_array(A, copy=True, ensure_2d=True)
        B = check_array(B, copy=True, ensure_2d=True)
        seeds_A = column_or_1d(seeds_A)
        seeds_B = column_or_1d(seeds_B)

        if A.shape[0] != B.shape[0]:
            msg = "Adjacency matrices must be of equal size"
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = "Adjacency matrix entries must be square"
            raise ValueError(msg)
        elif seeds_A.shape[0] != seeds_B.shape[0]:
            msg = "Seed arrays must be of equal size"
            raise ValueError(msg)
        elif seeds_A.shape[0] > A.shape[0]:
            msg = "There cannot be more seeds than there are nodes"
            raise ValueError(msg)
        elif not (seeds_A >= 0).all() or not (seeds_B >= 0).all():
            msg = "Seed array entries must be greater than or equal to zero"
            raise ValueError(msg)
        elif (
            not (seeds_A <= (A.shape[0] - 1)).all()
            or not (seeds_B <= (A.shape[0] - 1)).all()
        ):
            msg = "Seed array entries must be less than or equal to n-1"
            raise ValueError(msg)

        n = A.shape[0]  # number of vertices in graphs
        n_seeds = seeds_A.shape[0]  # number of seeds
        n_unseed = n - n_seeds

        score = math.inf
        perm_inds = np.zeros(n)

        obj_func_scalar = 1
        if self.gmp:
            obj_func_scalar = -1
            score = 0

        seeds_B_c = np.setdiff1d(range(n), seeds_B)
        if self.shuffle_input:
            seeds_B_c = np.random.permutation(seeds_B_c)
            # shuffle_input to avoid results from inputs that were already matched

        seeds_A_c = np.setdiff1d(range(n), seeds_A)
        permutation_A = np.concatenate([seeds_A, seeds_A_c], axis=None).astype(int)
        permutation_B = np.concatenate([seeds_B, seeds_B_c], axis=None).astype(int)
        A = A[np.ix_(permutation_A, permutation_A)]
        B = B[np.ix_(permutation_B, permutation_B)]

        # definitions according to Seeded Graph Matching [2].
        A11 = A[:n_seeds, :n_seeds]
        A12 = A[:n_seeds, n_seeds:]
        A21 = A[n_seeds:, :n_seeds]
        A22 = A[n_seeds:, n_seeds:]
        B11 = B[:n_seeds, :n_seeds]
        B12 = B[:n_seeds, n_seeds:]
        B21 = B[n_seeds:, :n_seeds]
        B22 = B[n_seeds:, n_seeds:]
        A11T = np.transpose(A11)
        A12T = np.transpose(A12)
        A22T = np.transpose(A22)
        B21T = np.transpose(B21)
        B22T = np.transpose(B22)

        for i in range(self.n_init):
            # setting initialization matrix
            if self.init_method == "rand":
                sk = SinkhornKnopp()
                K = np.random.rand(
                    n_unseed, n_unseed
                )  # generate a nxn matrix where each entry is a random integer [0,1]
                for i in range(10):  # perform 10 iterations of Sinkhorn balancing
                    K = sk.fit(K)
                J = np.ones((n_unseed, n_unseed)) / float(
                    n_unseed
                )  # initialize J, a doubly stochastic barycenter
                P = (K + J) / 2
            elif self.init_method == "barycenter":
                P = np.ones((n_unseed, n_unseed)) / float(n_unseed)

            const_sum = A21 @ np.transpose(B21) + np.transpose(A12) @ B12
            grad_P = math.inf  # gradient of P
            n_iter = 0  # number of FW iterations

            # OPTIMIZATION WHILE LOOP BEGINS
            while grad_P > self.eps and n_iter < self.max_iter:

                delta_f = (
                    const_sum + A22 @ P @ B22T + A22T @ P @ B22
                )  # computing the gradient of f(P) = -tr(APB^tP^t)
                rows, cols = linear_sum_assignment(
                    obj_func_scalar * delta_f
                )  # run hungarian algorithm on gradient(f(P))
                Q = np.zeros((n_unseed, n_unseed))
                Q[rows, cols] = 1  # initialize search direction matrix Q

                def f(x):  # computing the original optimization function
                    return obj_func_scalar * (
                        np.trace(A11T @ B11)
                        + np.trace(np.transpose(x * P + (1 - x) * Q) @ A21 @ B21T)
                        + np.trace(np.transpose(x * P + (1 - x) * Q) @ A12T @ B12)
                        + np.trace(
                            A22T
                            @ (x * P + (1 - x) * Q)
                            @ B22
                            @ np.transpose(x * P + (1 - x) * Q)
                        )
                    )

                alpha = minimize_scalar(
                    f, bounds=(0, 1), method="bounded"
                ).x  # computing the step size
                P_i1 = alpha * P + (1 - alpha) * Q  # Update P
                grad_P = np.linalg.norm(P - P_i1)
                P = P_i1
                n_iter += 1
            # end of FW optimization loop

            row, col = linear_sum_assignment(
                -P
            )  # Project onto the set of permutation matrices
            perm_inds_new = np.concatenate(
                (np.arange(n_seeds), np.array([x + n_seeds for x in col]))
            )

            score_new = np.trace(
                np.transpose(A) @ B[np.ix_(perm_inds_new, perm_inds_new)]
            )  # computing objective function value

            if obj_func_scalar * score_new < obj_func_scalar * score:  # minimizing
                score = score_new
                perm_inds = np.zeros(n, dtype=int)
                perm_inds[permutation_A] = permutation_B[perm_inds_new]

        permutation_A_unshuffle = _unshuffle(permutation_A, n)
        A = A[np.ix_(permutation_A_unshuffle, permutation_A_unshuffle)]
        permutation_B_unshuffle = _unshuffle(permutation_B, n)
        B = B[np.ix_(permutation_B_unshuffle, permutation_B_unshuffle)]
        score = np.trace(np.transpose(A) @ B[np.ix_(perm_inds, perm_inds)])

        self.perm_inds_ = perm_inds  # permutation indices
        self.score_ = score  # objective function value
        return self

    def fit_predict(self, A, B, seeds_A=[], seeds_B=[]):
        """
        Fits the model with two assigned adjacency matrices, returning optimal
        permutation indices

        Parameters
        ----------
        A : 2d-array, square, positive
            A square, positive adjacency matrix

        B : 2d-array, square, positive
            A square, positive adjacency matrix

        seeds_A : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `A`.

        seeds_B : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `B` The elements of
            `seeds_A` and `seeds_B` are vertices which are known to be matched, that is,
            `seeds_A[i]` is matched to vertex `seeds_B[i]`.

        Returns
        -------
        perm_inds_ : 1-d array, some shuffling of [0, n_vert)
            The optimal permutation indices to minimize the objective function
        """
        self.fit(A, B, seeds_A, seeds_B)
        return self.perm_inds_


def _unshuffle(array, n):
    unshuffle = np.array(range(n))
    unshuffle[array] = np.array(range(n))
    return unshuffle
