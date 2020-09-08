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

from scipy.linalg import orthogonal_procrustes

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
    `scipy.linalg.orthogonal_procrustes`, which itself uses an algorithm
    described in to find the optimal solution algorithm [1]_.

    Attributes
    ----------
        Q_ : array, size (d, d)
              final orthogonal matrix, used to modify X.

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    """

    def __init__(
        self,
    ):

        super().__init__()

    def fit(self, X, Y):
        """
        Uses the two datasets to learn the matrix Q_ that aligns the first
        dataset with the second.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions and entries as ones in Y.

        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            dimensions and entries as ones in Y.

        Returns
        -------
        self : returns an instance of self
        """
        X, Y = self._check_datasets(X, Y)

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
        self.Q_, _ = orthogonal_procrustes(X, Y)
        return self
