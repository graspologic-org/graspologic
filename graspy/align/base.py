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
from abc import abstractmethod
from sklearn.utils import check_array
from sklearn.base import BaseEstimator


class BaseAlign(BaseEstimator):
    """
    Base class for align tasks such as sign flipping, procrustes and seedless
    procrustes.

    Attributes
    ----------
        Q_ : array, size (d, d)
              final orthogonal matrix, used to modify X.

    """

    def __init__(self):
        pass

    def _check_datasets(self, X, Y):
        """
        Ensures that the datasets are numpy, 2d, finite, and have the same
        number of components. Does not check for same number of vertices.
        Returns copies of these datasets.
        """
        # check for numpy-ness
        if not isinstance(X, np.ndarray):
            msg = f"first dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        if not isinstance(Y, np.ndarray):
            msg = f"first dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        # check for 2-dness and finiteness
        X = check_array(X, copy=True)
        Y = check_array(Y, copy=True)
        # check for equal components
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)
        return X, Y

    @abstractmethod
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
        pass

    def transform(self, X):
        """
        Transforms the dataset X using the learned matrix Q_. This may be the
        same as the first dataset as in .fit(), or a new dataset. For example,
        additional samples from the same dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Dataset of vectors. Needs to have the same number of dimensions as
            X and Y passed to fit, but can have a different number of entries.

        Returns
        -------
        X_prime : np.ndarray, shape (n, d)
            First dataset of vectors, aligned to second. Equal to X @ self.Q_.
        """
        if not isinstance(X, np.ndarray):
            msg = f"dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X)
        return X @ self.Q_

    def fit_transform(self, X, Y):
        """
        Uses the two datasets to learn the matrix Q_ that aligns the first
        dataset with the second. Then, transforms the first dataset accordingly
        and returns it.

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
        X_prime : np.ndarray, shape (n, d)
            First dataset of vectors, aligned to second. Equal to X @ self.Q_X.
        """
        self.fit(X, Y)
        return self.transform(X)
