# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
from .qap import quadratic_assignment


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
        the FAQ algorithm will undergo. ``n_init`` automatically set to 1 if
        ``init_method`` = 'barycenter'

    init : string (default = 'barycenter') or 2d-array
        The initial position chosen

        If 2d-array, `init` must be :math:`m' x m'`, where :math:`m' = n - m`,
        and it must be doubly stochastic: each of its rows and columns must
        sum to 1.

        "barycenter" : the non-informative “flat doubly stochastic matrix,”
        :math:`J=1 \\times 1^T /n` , i.e the barycenter of the feasible region

        "rand" : some random point near :math:`J, (J+K)/2`, where K is some random
        doubly stochastic matrix

    max_iter : int, positive (default = 30)
        Integer specifying the max number of Franke-Wolfe iterations.
        FAQ typically converges with modest number of iterations.

    shuffle_input : bool (default = True)
        Gives users the option to shuffle the nodes of A matrix to avoid results
        from inputs that were already matched.

    eps : float (default = 0.1)
        A positive, threshold stopping criteria such that FW continues to iterate
        while Frobenius norm of :math:`(P_{i}-P_{i+1}) > \\text{eps}`

    gmp : bool (default = True)
        Gives users the option to solve QAP rather than the Graph Matching Problem
        (GMP). This is accomplished through trivial negation of the objective function.

    padding : string (default = 'adopted')
        Allows user to specify padding scheme if `A` and `B` are not of equal size.
        Say that `A` and `B` have :math:`n_1` and :math:`n_2` nodes, respectively, and
        :math:`n_1 < n_2`.

        "adopted" : matches `A` to the best fitting induced subgraph of `B`. Reduces the
        affinity between isolated vertices added to `A` through padding and low-density
        subgraphs of `B`.

        "naive" : matches `A` to the best fitting subgraph of `B`.

    Attributes
    ----------

    perm_inds_ : array, size (n,) where n is the number of vertices in the fitted graphs.
        The indices of the optimal permutation (with the fixed seeds given) on the nodes of B,
        to best minimize the objective function :math:`f(P) = \\text{trace}(A^T PBP^T )`.


    score_ : float
        The objective function value of for the optimal permutation found.

    n_iter_ : int
        Number of Frank-Wolfe iterations run. If ``n_init > 1``, :attr:`n_iter_` reflects the number of
        iterations performed at the initialization returned.


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
        init="barycenter",
        max_iter=30,
        shuffle_input=True,
        eps=0.1,
        gmp=True,
        padding="adopted",
    ):
        if type(n_init) is int and n_init > 0:
            self.n_init = n_init
        else:
            msg = '"n_init" must be a positive integer'
            raise TypeError(msg)
        if type(n_init) is int and n_init > 0:
            self.n_init = n_init
        else:
            msg = '"n_init" must be a positive integer'
            raise TypeError(msg)
        if init == "rand":
            self.init = "randomized"
        elif init == "barycenter":
            self.init = "barycenter"
        elif not isinstance(init, str):
            self.init = init
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
        if isinstance(padding, str) and padding in {"adopted", "naive"}:
            self.padding = padding
        elif isinstance(padding, str):
            msg = 'Invalid "padding" parameter string'
            raise ValueError(msg)
        else:
            msg = '"padding" parameter must be of type string'
            raise TypeError(msg)

    def fit(self, A, B, seeds_A=[], seeds_B=[]):
        """
        Fits the model with two assigned adjacency matrices

        Parameters
        ----------
        A : 2d-array, square
            A square adjacency matrix

        B : 2d-array, square
            A square adjacency matrix

        seeds_A : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in ``A``.

        seeds_B : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `B` The elements of
            ``seeds_A`` and ``seeds_B`` are vertices which are known to be matched, that is,
            ``seeds_A[i]`` is matched to vertex ``seeds_B[i]``.

        Returns
        -------
        self : returns an instance of self
        """
        A = check_array(A, copy=True, ensure_2d=True)
        B = check_array(B, copy=True, ensure_2d=True)
        seeds_A = column_or_1d(seeds_A)
        seeds_B = column_or_1d(seeds_B)
        partial_match = np.column_stack((seeds_A, seeds_B))

        # pads A and B according to section 2.5 of [2]
        if A.shape[0] != B.shape[0]:
            A, B = _adj_pad(A, B, self.padding)

        options = {
            "maximize": self.gmp,
            "partial_match": partial_match,
            "P0": self.init,
            "shuffle_input": self.shuffle_input,
            "maxiter": self.max_iter,
            "tol": self.eps,
        }

        res = min(
            [quadratic_assignment(A, B, options=options) for i in range(self.n_init)],
            key=lambda x: x.fun,
        )

        self.perm_inds_ = res.col_ind  # permutation indices
        self.score_ = res.fun  # objective function value
        self.n_iter_ = res.nit
        return self

    def fit_predict(self, A, B, seeds_A=[], seeds_B=[]):
        """
        Fits the model with two assigned adjacency matrices, returning optimal
        permutation indices

        Parameters
        ----------
        A : 2d-array, square
            A square adjacency matrix

        B : 2d-array, square
            A square adjacency matrix

        seeds_A : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in ``A``.

        seeds_B : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in ``B`` The elements of
            ``seeds_A`` and ``seeds_B`` are vertices which are known to be matched, that is,
            ``seeds_A[i]`` is matched to vertex ``seeds_B[i]``.

        Returns
        -------
        perm_inds_ : 1-d array, some shuffling of [0, n_vert)
            The optimal permutation indices to minimize the objective function
        """
        self.fit(A, B, seeds_A, seeds_B)
        return self.perm_inds_


def _adj_pad(A, B, method):
    def pad(X, n):
        X_pad = np.zeros((n[1], n[1]))
        X_pad[: n[0], : n[0]] = X
        return X_pad

    A_n = A.shape[0]
    B_n = B.shape[0]
    n = np.sort([A_n, B_n])
    if method == "adopted":
        A = 2 * A - np.ones((A_n, A_n))
        B = 2 * B - np.ones((B_n, B_n))

    if A.shape[0] == n[0]:
        A = pad(A, n)
    else:
        B = pad(B, n)

    return A, B
