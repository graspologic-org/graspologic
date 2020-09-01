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


class SignFlips(BaseAlign):
    """
    This module is used to transform two datasets by flipping the signs of all
    entries along some dimensions. In particular, it does so in a way that
    brings these two datasets to the same orthant according to some criteria,
    computed along each dimension. The two critera currently available are the
    median and the maximum value in magnitude along each dimension.

    There are two choices for which orthant the datasets end up in: either the
    first orthant (i.e. with all criteras being positive), which is a default,
    or and orthant in which one the second dataset is already in.

    This module can also run on a single dataset. In that case it makes the
    medians / maxes of each dimension of the provided dataset positive.

    Parameters
    ----------
        freeze_Y : boolean, optional (default False)
            For any two datasets X and Y, both of dimensions of d, there are
            2^d ways to bring their median / max signs to same orthant (one
            for each orthant). This flag chooses between one of the two options
            of how to pick such orthant.

            - True
                Acts only on matrix X (i.e. enforces Q_Y=I). In other words,
                changes the signs of X such that medians / maxes of all
                dimensions of X will match those of Y
            - False
                Makes medians / maxes of each dimension of both X and Y
                positive. In other words, brings the median / max of each
                dimension to the first orthant. Since in this case, choice of
                Q_X is independent of Y, it is not necessary to provide Y to
                fit at all.

        criteria : string, {'median' (default), 'max'}
            String describing the criteria used to choose whether to flip
            signs. Two options are currently supported:

            - 'median'
                uses the median along each dimension
            - 'max'
                uses the max (in magintude) alongs each dimension

    Attributes
    ----------
        Q_X : array, size (d, d)
              final diagonal orthogonal matrix, used to modify X

        Q_Y : array, size (d, d)
              final diagonal orthogonal matrix, used to modify Y

    """

    def __init__(
        self,
        freeze_Y=True,
        criteria="median",
    ):

        if type(criteria) is not str:
            raise TypeError("criteria must be str")
        if criteria not in ["median", "max"]:
            raise ValueError("{} is not a valid criteria.".format(criteria))

        super().__init__(freeze_Y=freeze_Y)

        self.criteria = criteria

    def set_criteria_function(self):
        # perform a check, in case it was modified directly
        if self.criteria not in ["median", "max"]:
            raise ValueError("{} is not a valid criteria".format(self.criteria))

        if self.criteria == "median":

            def median_criteria(X):
                return np.median(X, axis=0)

            self.criteria_function_ = median_criteria
        if self.criteria == "max":

            def max_criteria(X):
                return X[np.argmax(np.abs(X), axis=0), np.arange(X.shape[1])]

            self.criteria_function_ = max_criteria

    def fit(self, X, Y=None):
        """
        Uses the two datasets to learn matrices Q_X and Q_Y.
        In sign flips, Q_X and Q_Y are diagonal orthogonal matrices (i.e.
        matrices with 1 or -1 in each entry on diagonal and 0 everywhere else)
        picked such that all dimensions of X @ Q_X and Y @ Q_Y are in the same
        orthant using some critera (median or max magnitude).
        The second dataset can be omitted if freeze_Y was set to False; in that
        case Q_X is just the matrix that makes the medians / maxes of each
        dimension of X positive; only X_prime is returned in that case.

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y: np.ndarray, shape (m, d), or None
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.
            If freeze_Y is set to False, then it is appropriate to omit this,
            because X will just have all dimensions sign flipped to the first
            orthant anyway.

        Returns
        -------
        self: returns an instance of self
        """

        self.set_criteria_function()

        # check X for numpy-ness, dimensions and finiteness
        if not isinstance(X, np.ndarray):
            msg = (
                f"first dataset is a {type(X)}, not an np.ndarray! "
            )
            raise TypeError(msg)
        X = check_array(X, accept_sparse=True, copy=True)
        _, d = X.shape

        if Y is None:
            if self.freeze_Y:
                msg = (
                    "if freeze_Y=True, dataset X is matched to dataset Y. "
                    "hence, Y cannot be None. provide Y! (or set freeze_Y "
                    "to False, if you want to bring X to first orthant."
                )
                raise ValueError(msg)
            # make Y an identity as a filler so that we can use two matrix code
            Y = np.eye(d)

        # check for numpy-ness, 2d-ness and finite-ness
        if not isinstance(Y, np.ndarray):
            msg = (
                f"first dataset is a {type(Y)}, not an np.ndarray! "
            )
            raise TypeError(msg)
        Y = check_array(Y, accept_sparse=True, copy=True)

        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)

        X_criterias = self.criteria_function_(X)
        Y_criterias = self.criteria_function_(Y)

        if self.freeze_Y:
            val = np.multiply(X_criterias, Y_criterias)
            t_X = (val > 0) * 2 - 1
            t_Y = np.ones(d)
        else:
            t_X = np.sign(X_criterias).astype(int)
            t_Y = np.sign(Y_criterias).astype(int)

        self.Q_X, self.Q_Y = np.diag(t_X), np.diag(t_Y)
        return self

    def fit_transform(self, X, Y=None):
        """
        Learns the matrices Q_X and Q_Y, uses them to match the two datasets
        provided, and returns the two matched datasets.
        In sign flips, Q_X and Q_Y are diagonal orthogonal matrices (i.e.
        matrices with 1 or -1 in each entry on diagonal and 0 everywhere else)
        picked such that all dimensions of X @ Q_X and Y @ Q_Y are in the same
        orthant using some critera (median or max magnitude).
        The second dataset can be omitted if freeze_Y was set to False; in that
        case Q_X is just the matrix that makes the medians / maxes of each
        dimension of X positive; only X_prime is returned in that case.

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y: np.ndarray, shape (m, d), or None
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.
            If freeze_Y is set to False, then it is appropriate to omit this,
            because X will just have all dimensions sign flipped to the first
            orthant anyway.

        Returns
        -------
        X_prime: np.ndarray, shape (n, d)
            First dataset of vectors, matched to second. Equal to X @ self.Q_X.

        Y_prime: np.ndarray, shape (m, d)
            Second dataset of vectors, matched to first. Equal to X @ self.Q_Y.
            Unless Y was not provided - in that case only returns X_prime.
        """
        # SignFlips has an overloaded fit_transform, because unlike all other
        # aligners, it is appropriate to completely omit Y in fit.
        self.fit(X, Y)
        return self.transform(X, Y)
