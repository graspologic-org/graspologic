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
from functools import partial
from sklearn.utils import check_array

from .base import BaseAlign


class SignFlips(BaseAlign):
    """
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
                return X[np.arange(X.shape[0]) - 1, np.argmax(np.abs(X), axis=1)]

            self.criteria_function_ = max_criteria

    def fit(self, X, Y=None):
        # perform checks
        self.set_criteria_function()
        X = check_array(X, accept_sparse=True, copy=True)
        _, d = X.shape

        if Y is None:
            if self.freeze_Y:
                msg = "in fit, Y can only be None, if freeze_Y is False!"
                raise ValueError(msg)
            # make Y an identity as a filler so that we can use two matrix code
            Y = np.eye(d)

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
