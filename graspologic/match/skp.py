# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

# Title: sinkhorn_knopp Source Code
# Author: Tabanpour, B
# Date: 2018
# Code version:  0.2
# Availability: https://pypi.org/project/sinkhorn_knopp/
#
# The MIT License
#
# Copyright (c) 2016 Baruch Tabanpour
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import warnings
import numpy as np


class SinkhornKnopp:
    """
    Sinkhorn Knopp Algorithm
    Takes a non-negative square matrix P, where P =/= 0
    and iterates through Sinkhorn Knopp's algorithm
    to convert P to a doubly stochastic matrix.
    Guaranteed convergence if P has total support [1]:

    Parameters
    ----------
    max_iter : int, default=1000
        The maximum number of iterations.

    epsilon : float, default=1e-3
        Metric used to compute the stopping condition,
        which occurs if all the row and column sums are
        within epsilon of 1. This should be a very small value.
        Epsilon must be between 0 and 1.

    Attributes
    ----------
    _max_iter : int, default=1000
        User defined parameter. See above.

    _epsilon : float, default=1e-3
        User defined paramter. See above.

    _stopping_condition: string
        Either "max_iter", "epsilon", or None, which is a
        description of why the algorithm stopped iterating.

    _iterations : int
        The number of iterations elapsed during the algorithm's
        run-time.

    _D1 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.

    _D2 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.


    References
    ----------
    .. [1] Sinkhorn, Richard & Knopp, Paul. (1967). "Concerning nonnegative
           matrices and doubly stochastic matrices," Pacific Journal of
           Mathematics. 21. 10.2140/pjm.1967.21.343.
    """

    def __init__(self, max_iter=1000, epsilon=1e-3):
        if type(max_iter) is int or type(max_iter) is float:
            if max_iter > 0:
                self._max_iter = int(max_iter)
            else:
                msg = "max_iter must be greater than 0"
                raise ValueError(msg)
        else:
            msg = "max_iter is not of type int or float"
            raise TypeError(msg)

        if type(epsilon) is int or type(epsilon) is float:
            if epsilon > 0 and epsilon < 1:
                self._epsilon = int(epsilon)
            else:
                msg = "epsilon must be between 0 and 1 exclusively"
                raise ValueError(msg)
        else:
            msg = "epsilon is not of type int or float"
            raise TypeError(msg)

        self._stopping_condition = None
        self._iterations = 0
        self._D1 = np.ones(1)
        self._D2 = np.ones(1)

    def fit(self, P):
        """
        Fit the diagonal matrices in Sinkhorn Knopp's algorithm

        Parameters
        ----------
        P : 2d array-like
            Must be a square non-negative 2d array-like object, that
            is convertible to a numpy array. The matrix must not be
            equal to 0 and it must have total support for the algorithm
            to converge.

        Returns
        -------
        P_eps : A double stochastic matrix.
        """
        P = np.asarray(P)
        assert np.all(P >= 0)
        assert P.ndim == 2
        assert P.shape[0] == P.shape[1]

        N = P.shape[0]
        max_thresh = 1 + self._epsilon
        min_thresh = 1 - self._epsilon

        # Initialize r and c, the diagonals of D1 and D2
        # and warn if the matrix does not have support.
        r = np.ones((N, 1))
        pdotr = P.T.dot(r)
        total_support_warning_str = (
            "Matrix P must have total support. " "See documentation"
        )
        if not np.all(pdotr != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        c = 1 / pdotr
        pdotc = P.dot(c)
        if not np.all(pdotc != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        r = 1 / pdotc
        del pdotr, pdotc

        P_eps = np.copy(P)
        while (
            np.any(np.sum(P_eps, axis=1) < min_thresh)
            or np.any(np.sum(P_eps, axis=1) > max_thresh)
            or np.any(np.sum(P_eps, axis=0) < min_thresh)
            or np.any(np.sum(P_eps, axis=0) > max_thresh)
        ):

            c = 1 / P.T.dot(r)
            r = 1 / P.dot(c)

            self._D1 = np.diag(np.squeeze(r))
            self._D2 = np.diag(np.squeeze(c))
            P_eps = self._D1.dot(P).dot(self._D2)

            self._iterations += 1

            if self._iterations >= self._max_iter:
                self._stopping_condition = "max_iter"
                break

        if not self._stopping_condition:
            self._stopping_condition = "epsilon"

        self._D1 = np.diag(np.squeeze(r))
        self._D2 = np.diag(np.squeeze(c))
        P_eps = self._D1.dot(P).dot(self._D2)

        return P_eps
