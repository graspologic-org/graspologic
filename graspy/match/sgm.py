# Copyright 2019 NeuroData (http://neurodata.io)
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
import math
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from sklearn.utils import check_array
from .skp import SinkhornKnopp


class SeededGraphMatching:
    """
    Seeded Graph Matching Algorithm (SGM)
    The seeded graph matching problem is a variation of graph matching
     in which part of the matching is fixed. This algorithm is a modification
     of FAQ, an algorithm also implemented in this package[1].


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
        Integer specifying the max number of FW iterations.
        FAQ typically converges with modest number of iterations

    shuffle_input : bool (default = True)
        Gives users the option to shuffle the nodes of A matrix to avoid results
        from inputs that were already matched

    eps : float (default = 0.1)
        A positive, threshold stopping criteria such that FW continues to iterate
        while Frobenius norm of :math:`(P_{i}-P_{i+1}) > eps`


    gmp : bool (default = False)
        Gives users the option to the Graph Matching Problem (GMP) rather than
        the Quadratic Assignment (QAP). This is accomplished through trivial
        negation of the objective function.

    Attributes
    ----------

    perm_inds_ : array, size (n,) where n is the number of vertices in the graphs fitted.
        The indices of the optimal permutation on the nodes of B, found via
        FAQ, to best minimize the objective function :math:`f(P) = trace(A^T PBP^T )`.


    score_ : float
        The objective function value of for the optimal permutation found.


    References
    ----------
    .. [1] J. T. Vogelstein, J. M. Conroy, V. Lyzinski, L. J. Podrazik, S. G. Kratzer,
           E. T. Harley, D. E. Fishkind, R. J. Vogelstein, and C. E. Priebe, “Fast
           approximate quadratic programming for graph matching,” PLOS one, vol. 10,
           no. 4, p. e0121002, 2015.



    """

    def __init__(
        self,
        n_init=1,
        init_method="barycenter",
        max_iter=30,
        shuffle_input=True,
        eps=0.1,
        gmp=False,
    ):

        if n_init > 0 and type(n_init) is int:
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

    def fit(self, A, B, W1, W2):
        """
        Fits the model with two assigned adjacency matrices

        Parameters
        ---------
        A : 2d-array, square, positive
            A square, positive adjacency matrix

        B : 2d-array, square, positive
            A square, positive adjacency matrix

        W1 : 1d-array, shape (m , 1) where m <= n
            An array where each entry is a node in A

        W2 : 1d-array, shape (m , 1) where m <= n
            An array where each entry is a node in B
            The elements of W1 and W2 are seeds, creating a fixed
            seeding of W1 -> W2

        Returns
        -------

        self : returns an instance of self
        """
        A = check_array(A, copy=True, ensure_2d=True)
        B = check_array(B, copy=True, ensure_2d=True)

        if A.shape[0] != B.shape[0]:
            msg = "Adjacency matrices must be of equal size"
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = "Adjacency matrix entries must be square"
            raise ValueError(msg)
        elif (A >= 0).all() == False or (B >= 0).all() == False:
            msg = "Adjacency matrix entries must be greater than or equal to zero"
            raise ValueError(msg)
        elif W1.shape[0] != W2.shape[0]:
            msg = "Seed arrays must be of equal size"
            raise ValueError(msg)
        elif W1.shape[0] > A.shape[0]:
            msg = "There cannot be more seeds than there are nodes"
            raise ValueError(msg)
        elif (W1 >= 0).all() == False or (W2 >= 0).all() == False:
            msg = "Seed array entries must be greater than or equal to zero"
            raise ValueError(msg)

        n = A.shape[0]  # number of vertices in graphs
        n_seeds = W1.shape[0]  # number of seeds
        n_unseed = n - n_seeds

        obj_func_scalar = 1
        if self.gmp:
            obj_func_scalar = -1

        p_A = np.concatenate(
            [W1, np.array([x for x in range(n) if x not in W1])], axis=None
        )
        p_B = np.concatenate(
            [W2, np.array([x for x in range(n) if x not in W2])], axis=None
        )
        A = A[np.ix_(p_A, p_A)]
        B = B[np.ix_(p_B, p_B)]
        if self.shuffle_input:
            node_shuffle_input = np.concatenate(
                (np.arange(n_seeds), np.random.permutation(np.arange(n_seeds, n)))
            )
            A = A[np.ix_(node_shuffle_input, node_shuffle_input)]
            # shuffle_input to avoid results from inputs that were already matched

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

        score = math.inf
        perm_inds = np.zeros(n)

        for i in range(self.n_init):

            # setting initialization matrix
            if self.init_method == "rand":
                sk = SinkhornKnopp()
                K = np.random.rand(
                    n - n_seeds, n - n_seeds
                )  # generate a nxn matrix where each entry is a random integer [0,1]
                for i in range(10):  # perform 10 iterations of Sinkhorn balancing
                    K = sk.fit(K)
                J = np.ones((n - n_seeds, n - n_seeds)) / float(
                    n - n_seeds
                )  # initialize J, a doubly stochastic barycenter
                P = (K + J) / 2
            elif self.init_method == "barycenter":
                P = np.ones((n - n_seeds, n - n_seeds)) / float(n - n_seeds)

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
                Q = np.zeros((n, n))
                Q[rows, cols] = 1  # initialize search direction matrix Q

                def f(x):  # computing the original optimization function
                    return obj_func_scalar * np.trace(
                        np.transpose(A)
                        @ np.stack(
                            (
                                np.hstack((B11, B12 @ np.transpose(P))),
                                np.hstack((P @ B21, P @ B22 @ np.transpose(P))),
                            )
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
                if self.shuffle_input:
                    perm_inds = np.array([0] * n)
                    perm_inds[node_shuffle_input] = perm_inds_new
                    perm_inds[p_A] = perm_inds
                else:
                    perm_inds[p_A] = perm_inds_new

        if self.shuffle_input:
            node_unshuffle_input = np.array(range(n))
            node_unshuffle_input[node_shuffle_input] = np.array(range(n))
            A = A[np.ix_(node_unshuffle_input, node_unshuffle_input)]
            score = np.trace(np.transpose(A) @ B[np.ix_(perm_inds, perm_inds)])

        p_A_unshuffle = np.array(range(n))
        p_B_unshuffle = np.array(range(n))
        p_A_unshuffle[p_A] = np.array(range(n))
        p_B_unshuffle[p_B] = np.array(range(n))
        A = A[np.ix_(p_A_unshuffle, p_A_unshuffle)]
        B = B[np.ix_(p_B_unshuffle, p_B_unshuffle)]
        score = np.trace(np.transpose(A) @ B[np.ix_(perm_inds, perm_inds)])

        self.perm_inds_ = perm_inds  # permutation indices
        self.score_ = score  # objective function value
        return self

    def fit_predict(self, A, B):
        """
        Fits the model with two assigned adjacency matrices, returning optimal
        permutation indices

        Parameters
        ---------
        A : 2d-array, square, positive
            A square, positive adjacency matrix

        B : 2d-array, square, positive
            A square, positive adjacency matrix

        Returns
        -------

        perm_inds_ : 1-d array, some shuffling of [0, n_vert)
            The optimal permutation indices to minimize the objective function
        """
        self.fit(A, B)
        return self.perm_inds_
