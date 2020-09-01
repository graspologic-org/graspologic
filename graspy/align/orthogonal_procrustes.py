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
from sklearn.linalg import orthogonal_procrustes

from .base import BaseAlign


class OrthogonalProcrustes(BaseAlign):
    """
    Compute the matrix solution of the orthogonal Procrustes problem, which is
    that given two matrices X and Y of equal shape, find an orthogonal matrix
    that most closely maps X to Y.

    Note that when used to match two datasets, this method unlike
    SeedlessProcrustes, not only requires that the datasets have the same
    number of entries, but also that there is some correspondence between the
    vertices. In statistical spectral graphs, this usually corresponds to the
    assumption that the vertex i in graph X has the same latent position as the
    vertex i in graph Y.

    Implementation-wise, this class is a wrapper of the
    `sklearn.linalg.orthogonal_procrustes`, which itself uses an algorithm
    described in to find the optimal solution algorithm [1]_.

    Parameters
    ----------
        freeze_Y : boolean, optional (default True)
            Irrelevant in OrthogonalProcrustes, as it always modifies only one
            dataset. Exists for compatibility with other align modules.

    Attributes
    ----------
        Q_X : array, size (d, d)
              final orthogonal matrix, used to modify X

        Q_Y : array, size (d, d)
              final orthogonal matrix, used to modify Y
              in OrthogonalProcrustes Q_Y is always equal to identity I

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    """

    def __init__(
        self,
        freeze_Y=True,
    ):
        super().__init__(freeze_Y=freeze_Y)

    def fit(self, X, Y):
        """
        Uses the two datasets to learn matrices Q_X and Q_Y.
        In regular orthogonal procrustes Q_X is a solution to the orthogonal
        procrustes problem, and Q_Y is the identity matrix.

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
        if X.shape[0] != Y.shape[0]:
            msg = (
                "two datasets have different number of entries! "
                "OrthogonalProcrustes assumes that entries of the two "
                "datasets are matched. consider using SeedlessProcrustes "
                "instead."
            )
            raise ValueError(msg)
        _, d = X.shape

        # call scipy's orthogonal procrustes, set the second matrix to identity
        self.Q_X, _ = orthogonal_procrustes(X, Y)
        self.Q_Y = np.eye(d)

        return self
