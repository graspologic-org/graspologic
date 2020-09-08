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
        criteria : string, {'median' (default), 'max'}, optional
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
        criteria="median",
    ):
        # checking criteria argument
        if type(criteria) is not str:
            raise TypeError("criteria must be str")
        if criteria not in ["median", "max"]:
            raise ValueError("{} is not a valid criteria.".format(criteria))

        super().__init__()

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

    def fit(self, X, Y):
        """
        Uses the two datasets to learn matrices Q_X and Q_Y.
        In sign flips, Q_X and Q_Y are diagonal orthogonal matrices (i.e.
        matrices with 1 or -1 in each entry on diagonal and 0 everywhere else)
        picked such that all dimensions of X @ Q_X and Y @ Q_Y are in the same
        orthant using some critera (median or max magnitude).

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y: np.ndarray, shape (m, d), or None
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.

        Returns
        -------
        self: returns an instance of self
        """
        # check X for numpy-ness, dimensions and finiteness
        if not isinstance(X, np.ndarray):
            msg = f"first dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X, copy=True)
        # check for numpy-ness, 2d-ness and finite-ness
        if not isinstance(Y, np.ndarray):
            msg = f"first dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        Y = check_array(Y, copy=True)
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)
        _, d = X.shape

        self.set_criteria_function()

        X_criterias = self.criteria_function_(X)
        Y_criterias = self.criteria_function_(Y)

        val = np.multiply(X_criterias, Y_criterias)
        t_X = (val > 0) * 2 - 1
        t_Y = np.ones(d)

        self.Q_X, self.Q_Y = np.diag(t_X), np.diag(t_Y)
        return self
